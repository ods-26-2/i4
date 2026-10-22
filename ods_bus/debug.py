"""
ods_bus/debug.py
=================
Ferramentas de teste do barramento.

1) Ver o que está chegando e se está conforme o contrato:
    python -m ods_bus.debug tail ods.i4.sign ods.i4.keypoints --max 40 --timeout 60
    (acrescente --from-start para ler o histórico do tópico)

2) Simular o A6 pedindo uma síntese ao S6 e conferir a reply:
    python -m ods_bus.debug synth ALUNO BANCO --lexicon-version lexicon-a6-1.3

Os dois terminam sozinhos (--max / --timeout), pra não gastar créditos do GCP.
Cada execução usa um group.id novo, então não atrapalha o offset de ninguém.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

from ods_bus.client import BusProducer, bootstrap_servers, reply_schema, topic_for
from ods_bus.envelope import build_message, validate_message
from ods_bus.client import ensure_topics


def _check_keypoints(p: dict) -> list[str]:
    problems = []
    if len(p.get("body", [])) not in (0, 33):
        problems.append(f"body tem {len(p['body'])} pontos (esperado 0 ou 33)")
    for name, n in (("left_hand", 21), ("right_hand", 21), ("face", 92)):
        v = p.get(name)
        if v is not None and len(v) != n:
            problems.append(f"{name} tem {len(v)} pontos (esperado {n} ou null)")
    return problems


def _consumer(offset: str):
    from confluent_kafka import Consumer

    return Consumer(
        {
            "bootstrap.servers": bootstrap_servers(),
            "group.id": f"debug-{uuid.uuid4().hex[:8]}",
            "auto.offset.reset": offset,
        }
    )


def tail(args) -> None:
    consumer = _consumer("earliest" if args.from_start else "latest")
    consumer.subscribe([topic_for(s) for s in args.schemas])
    last_seq: dict[tuple, int] = {}
    seqs: dict[tuple, set[int]] = {}
    seen = bad = 0
    deadline = time.time() + args.timeout
    print(f"ouvindo {args.schemas} em {bootstrap_servers()} (até {args.max} msgs ou {args.timeout:.0f}s)")
    try:
        while seen < args.max and time.time() < deadline:
            m = consumer.poll(1.0)
            if m is None or m.error():
                continue
            seen += 1
            try:
                msg = json.loads(m.value().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                bad += 1
                print(f"[{seen}] INVÁLIDA: não é JSON")
                continue
            ok, why = validate_message(msg)
            problems = [] if ok else [why]
            if ok:
                if msg["schema"] != m.topic() and topic_for(msg["schema"]) != m.topic():
                    problems.append(f"schema {msg['schema']} em tópico {m.topic()}")
                if msg["schema"] == "ods.i4.keypoints":
                    problems += _check_keypoints(msg["payload"])
                # if "sequence" in msg:  # detecta perda/reordenação por produtor+sessão
                #     key = (msg["producer"], msg.get("session_id"))
                #     prev = last_seq.get(key)
                #     if prev is not None and msg["sequence"] != prev + 1:
                #         problems.append(f"sequence pulou de {prev} para {msg['sequence']}")
                #     last_seq[key] = msg["sequence"]
                if "sequence" in msg:
                    key = (msg["producer"], msg.get("session_id"))
                    if msg["sequence"] in seqs.setdefault(key, set()):
                        problems.append(f"sequence {msg['sequence']} repetida")
                    seqs[key].add(msg["sequence"])
                    
            bad += bool(problems)
            status = "OK " if not problems else "ERRO"
            print(
                f"[{seen}] {status} {msg.get('schema')} prod={msg.get('producer')} "
                f"seq={msg.get('sequence')} cap={msg.get('captured_at')} "
                f"payload={list(msg.get('payload', {}))[:6]}"
            )
            for p in problems:
                print(f"        ! {p}")
    finally:
        consumer.close()
        for (prod, sess), s in seqs.items():
            missing = sorted(set(range(min(s), max(s) + 1)) - s)
            if missing:
                print(f"! {prod}/{sess}: sequences faltando: {missing[:10]}")
    print(f"\n{seen} mensagens, {bad} com problema")


def synth(args) -> None:
    schema = "ods.s6.synthesis.command"
    consumer = _consumer("earliest")  # earliest: não perde a reply por corrida de assinatura
    
    ensure_topics([topic_for(schema), topic_for(reply_schema(schema))])
    consumer.subscribe([topic_for(reply_schema(schema))])
    for _ in range(5):  # deixa o consumidor receber as partições antes de enviar
        consumer.poll(0.5)

    cmd = build_message(
        message_type="command",
        schema=schema,
        producer="A6",
        profile="CMD",
        target="S6",
        payload={
            "glosses": args.glosses,
            "speed_factor": args.speed,
            "lexicon_version": args.lexicon_version,
            **({"markers": [{"marker": args.marker, "from_gloss_index": 0, "to_gloss_index": len(args.glosses) - 1}]} if args.marker else {}),
        },
    )
    producer = BusProducer()
    producer.publish(cmd, key=cmd["message_id"])
    producer.flush()
    print(f"comando enviado: {cmd['message_id']}")

    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline:
            m = consumer.poll(1.0)
            if m is None or m.error():
                continue
            msg = json.loads(m.value().decode("utf-8"))
            if msg.get("correlation_id") != cmd["message_id"]:
                continue
            ok, why = validate_message(msg, schema=reply_schema(schema), profile="RPL")
            print("envelope da reply:", "OK" if ok else f"INVÁLIDO ({why})")
            print(json.dumps(msg["payload"], indent=2, ensure_ascii=False))
            return
        print(f"sem reply em {args.timeout:.0f}s — o serviço do S6 está rodando? (python -m s6_synthesis.bus_service)")
    finally:
        consumer.close()


def main():
    ap = argparse.ArgumentParser(description="Ferramentas de teste do barramento ODS.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tail", help="ouve tópicos e valida as mensagens")
    t.add_argument("schemas", nargs="+")
    t.add_argument("--max", type=int, default=50)
    t.add_argument("--timeout", type=float, default=60)
    t.add_argument("--from-start", action="store_true")
    t.set_defaults(fn=tail)

    s = sub.add_parser("synth", help="simula o A6 pedindo síntese ao S6")
    s.add_argument("glosses", nargs="+")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--lexicon-version", default="lexicon-a6-1.3")
    s.add_argument("--marker", default=None, help="ex.: INTERROGATIVA_SN (cobre todas as glosas)")
    s.add_argument("--timeout", type=float, default=20)
    s.set_defaults(fn=synth)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()