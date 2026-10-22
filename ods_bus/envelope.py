"""
ods_bus/envelope.py
====================
Envelope comum das mensagens (Especificação de interfaces 1.0, seção 3) e
validação no consumo (guia do Kafka, seção 4.3).

Não depende de Kafka: dá pra importar e testar sem o cliente instalado.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

SCHEMA_VERSION = "1.0"
SUPPORTED_MAJOR = "1"

# Os seis campos fixos de toda mensagem.
FIXED_FIELDS = ("message_type", "message_id", "schema", "schema_version", "producer", "published_at")

# Campos obrigatórios de cada perfil, além dos seis fixos.
PROFILE_REQUIRED: dict[str, tuple[str, ...]] = {
    "EV-SINAL": ("captured_at", "source_id", "session_id", "sequence", "artifact_version"),
    "EV-SERVICO": ("captured_at", "session_id", "sequence"),
    "CMD": ("target",),
    "RPL": ("correlation_id",),
}


# ------------------------------------------------------------------ tempo

def utc_iso(dt: Optional[datetime] = None) -> str:
    """ISO 8601 em UTC com milissegundos: 2026-09-23T14:29:59.870Z."""
    dt = (dt or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def epoch_ms_to_iso(ms: int) -> str:
    """Epoch em ms (como o server.cpp grava em ts_ms) -> ISO 8601 UTC."""
    base = datetime.fromtimestamp(int(ms) // 1000, tz=timezone.utc)
    return utc_iso(base + timedelta(milliseconds=int(ms) % 1000))


# --------------------------------------------------------------- sequence

class SequenceCounter:
    """`sequence`: contador crescente por producer + session_id, a partir de 0."""

    def __init__(self) -> None:
        self._n: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def next(self, producer: str, session_id: str) -> int:
        with self._lock:
            n = self._n.get((producer, session_id), 0)
            self._n[(producer, session_id)] = n + 1
            return n


# ------------------------------------------------------------- construção

def build_message(
    *,
    message_type: str,
    schema: str,
    producer: str,
    payload: dict[str, Any],
    profile: str,
    **context: Any,
) -> dict[str, Any]:
    """Monta o envelope. `context` leva os campos do perfil (captured_at,
    source_id, session_id, sequence, artifact_version, target,
    correlation_id). Campos None são omitidos; faltando algum que o perfil
    exige, levanta ValueError (melhor falhar no publicador, como manda B1)."""
    msg: dict[str, Any] = {
        "message_type": message_type,
        "message_id": str(uuid.uuid4()),
        "schema": schema,
        "schema_version": SCHEMA_VERSION,
        "producer": producer,
        "published_at": utc_iso(),
    }
    for key, value in context.items():
        if value is not None:
            msg[key] = value
    missing = [f for f in PROFILE_REQUIRED[profile] if f not in msg]
    if missing:
        raise ValueError(f"{schema}: perfil {profile} exige {missing}")
    msg["payload"] = payload
    return msg


# -------------------------------------------------------------- validação

def validate_message(
    msg: Any, *, schema: Optional[str] = None, profile: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """(True, None) se o envelope está de acordo; senão (False, motivo).
    A mensagem inválida deve ser descartada e o motivo logado."""
    if not isinstance(msg, dict):
        return False, "mensagem não é um objeto JSON"
    for field in (*FIXED_FIELDS, "payload"):
        if field not in msg:
            return False, f"campo obrigatório ausente: {field}"
    if schema is not None and msg["schema"] != schema:
        return False, f"schema inesperado: {msg['schema']}"
    if str(msg["schema_version"]).split(".")[0] != SUPPORTED_MAJOR:
        return False, f"versão não suportada: {msg['schema_version']}"
    if not isinstance(msg["payload"], dict):
        return False, "payload não é um objeto"
    if profile is not None:
        for field in PROFILE_REQUIRED[profile]:
            if field not in msg:
                return False, f"perfil {profile}: campo ausente: {field}"
    return True, None