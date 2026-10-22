"""
ods_bus/client.py
==================
Produtor e serviço de comandos sobre o Kafka do projeto (guia do TRA-1).

- O endereço do broker NÃO é fixo no código: vem de ODS_KAFKA_BOOTSTRAP
  (padrão = instância de teste na nuvem). Quando o serviço passar a rodar
  localmente, basta trocar a variável.
- Cuidado com créditos do GCP (guia, seção 4.5): nada aqui faz polling
  apertado (poll bloqueia até 1 s), e o serviço encerra limpo em Ctrl+C.

Requisito: pip install confluent-kafka
"""

from __future__ import annotations

import json
import logging
import os
import signal
from collections import OrderedDict
from typing import Any, Callable, Optional

from ods_bus.envelope import build_message, validate_message

log = logging.getLogger("ods_bus")

DEFAULT_BOOTSTRAP = "34.66.235.91:9093"  # TEMPORÁRIO (nuvem, só testes)

# Tópico = schema (ods.i4.sign). O guia do Kafka sugere nomes por domínio
# (ods.gesto.reconhecido); se o TRA-1 decidir por eles, é só mapear aqui:
#   TOPIC_OVERRIDES["ods.i4.sign"] = "ods.gesto.reconhecido"
TOPIC_OVERRIDES: dict[str, str] = {}


def bootstrap_servers() -> str:
    return os.environ.get("ODS_KAFKA_BOOTSTRAP", DEFAULT_BOOTSTRAP)


def topic_for(schema: str) -> str:
    return TOPIC_OVERRIDES.get(schema, schema)


def reply_schema(schema: str) -> str:
    """ods.s6.synthesis.command -> ods.s6.synthesis.reply (idem para .query)."""
    return schema.rsplit(".", 1)[0] + ".reply"


class BusError(Exception):
    """Erro de negócio de um handler; vira reply com status ERROR."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        
def ensure_topics(topics, bootstrap=None, partitions=1, replication=1):
    """Cria os tópicos que ainda não existem (evita a corrida com o auto-create)."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap or bootstrap_servers()})
    existing = set(admin.list_topics(timeout=10).topics)
    missing = [t for t in topics if t not in existing]
    if not missing:
        return
    futures = admin.create_topics([NewTopic(t, partitions, replication) for t in missing])
    for topic, fut in futures.items():
        try:
            fut.result()
            log.info("tópico criado: %s", topic)
        except Exception as exc:
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                log.warning("não consegui criar %s: %s", topic, exc)


# ---------------------------------------------------------------- produtor

class BusProducer:
    def __init__(self, bootstrap: Optional[str] = None, extra_conf: Optional[dict] = None):
        from confluent_kafka import Producer

        conf = {"bootstrap.servers": bootstrap or bootstrap_servers(), "linger.ms": 20}
        conf.update(extra_conf or {})
        self._p = Producer(conf)

    @staticmethod
    def _on_delivery(err, msg) -> None:
        if err is not None:
            log.error("falha ao entregar em %s: %s", msg.topic(), err)

    def publish(self, msg: dict[str, Any], key: Optional[str] = None) -> None:
        """Publica no tópico do schema. A chave padrão é o session_id, que
        mantém as mensagens de uma sessão na mesma partição (ordenadas)."""
        key = key or msg.get("session_id") or msg.get("correlation_id")
        value = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        kwargs = dict(
            topic=topic_for(msg["schema"]),
            value=value,
            key=key.encode("utf-8") if key else None,
            on_delivery=self._on_delivery,
        )
        try:
            self._p.produce(**kwargs)
        except BufferError:  # fila interna cheia: deixa escoar e tenta de novo
            self._p.poll(0.5)
            self._p.produce(**kwargs)
        self._p.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        remaining = self._p.flush(timeout)
        if remaining:
            log.warning("%d mensagem(ns) não entregue(s) após %.0fs", remaining, timeout)

    close = flush


# ------------------------------------------------------ serviço de comandos

Handler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class CommandService:
    """Atende comandos/consultas endereçados a `component` e publica a reply.

    handlers: {schema_do_comando: fn(payload, msg) -> dict_do_payload_da_reply}.
    O handler devolve só os campos específicos; o `status: OK` é acrescentado
    aqui. Para falhar de forma controlada, levanta BusError(code, message).
    """

    def __init__(
        self,
        component: str,
        group_id: str,
        handlers: dict[str, Handler],
        producer: BusProducer,
        bootstrap: Optional[str] = None,
        auto_offset_reset: str = "latest",
        reply_cache_size: int = 256,
    ):
        from confluent_kafka import Consumer

        self.component = component
        self.handlers = handlers
        self.producer = producer
        self._stop = False
        # Comandos já atendidos: se o mesmo message_id chegar de novo (A6
        # reenviou porque perdeu a reply), reenvia a MESMA reply em vez de
        # refazer o trabalho (idempotência pelo message_id).
        self._replies: OrderedDict[str, dict] = OrderedDict()
        self._reply_cache_size = reply_cache_size
        # "latest" por padrão: um group.id novo não reprocessa o histórico de
        # comandos antigos do tópico.
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap or bootstrap_servers(),
                "group.id": group_id,
                "auto.offset.reset": auto_offset_reset,
                "topic.metadata.refresh.interval.ms": 10000,
            }
        )

    def stop(self, *_: Any) -> None:
        self._stop = True

    def run(self) -> None:
        from confluent_kafka import KafkaError

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
        topics = [topic_for(s) for s in self.handlers]
        topics += [topic_for(reply_schema(s)) for s in self.handlers]
        ensure_topics(topics)
        self._consumer.subscribe([topic_for(s) for s in self.handlers])
        
        self._consumer.subscribe([topic_for(s) for s in self.handlers])
        log.info("%s ouvindo %s", self.component, sorted(self.handlers))
        try:
            while not self._stop:
                msg = self._consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        log.warning("erro de leitura: %s", msg.error())
                    continue
                self.handle_raw(msg.value())
        finally:
            self._consumer.close()
            self.producer.flush()
            log.info("%s encerrado", self.component)

    def handle_raw(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("mensagem descartada: JSON inválido")
            return
        ok, why = validate_message(msg, profile="CMD")
        if not ok:
            log.warning("mensagem descartada: %s", why)
            return
        handler = self.handlers.get(msg["schema"])
        if handler is None or msg["target"] != self.component:
            return  # não é para nós

        mid = msg["message_id"]
        if mid in self._replies:
            log.info("comando duplicado %s: reenviando a reply anterior", mid)
            self.producer.publish(self._replies[mid], key=mid)
            return

        try:
            payload = {"status": "OK", **handler(msg["payload"], msg)}
        except BusError as exc:
            payload = {"status": "ERROR", "error": {"code": exc.code, "message": exc.message}}
        except Exception:
            log.exception("erro inesperado em %s", msg["schema"])
            payload = {"status": "ERROR", "error": {"code": "INTERNAL_ERROR", "message": "falha interna"}}

        reply = build_message(
            message_type="reply",
            schema=reply_schema(msg["schema"]),
            producer=self.component,
            payload=payload,
            profile="RPL",
            correlation_id=mid,
        )
        self._replies[mid] = reply
        while len(self._replies) > self._reply_cache_size:
            self._replies.popitem(last=False)
        self.producer.publish(reply, key=mid)
        self.producer.flush(5.0)