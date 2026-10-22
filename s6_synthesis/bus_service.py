"""
s6_synthesis/bus_service.py
=============================
S6 no barramento: atende `ods.s6.synthesis.command` (vindo de A6) e publica
`ods.s6.synthesis.reply` com correlation_id = message_id do comando.

A trajetória NÃO vai na reply (convenção: conteúdo grande vai por URI). O S6
grava o KEYPOINT_SEQUENCE num arquivo e devolve `trajectory_ref`.

Uso:
    python -m s6_synthesis.bus_service --lexicon-dir lexicon \
        --trajectory-dir trajectories --lexicon-version lexicon-a6-1.3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from ods_bus.client import BusError, BusProducer, CommandService
from ods_bus.keypoints import to_spec_frame
from s6_synthesis.s6_pipeline import S6SynthesisPipeline
from s6_synthesis.synthesis.synthesizer import synthesize

COMPONENT = "S6"
COMMAND_SCHEMA = "ods.s6.synthesis.command"

# Conjunto fechado de marcadores (exemplo do artefato LEXICON na spec, seção 8).
# Quando existir o artefato LEXICON publicado por A6, ler daí.
DEFAULT_MARKERS = frozenset(
    {"INTERROGATIVA_QU", "INTERROGATIVA_SN", "NEGACAO", "AFIRMACAO", "NEUTRO"}
)


def _norm(text: str) -> str:
    """Compara glosas sem depender de caixa nem de acento: 'Aluno' == 'ALUNO'."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").upper().strip()


class TrajectoryStore:
    """Grava o KEYPOINT_SEQUENCE em disco e devolve a URI s6://trajectories/...
    Quem consome (A6) precisa enxergar essa pasta (volume compartilhado) até
    existir um registro de artefatos (B5)."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict[str, Any]) -> str:
        name = f"{uuid.uuid4().hex[:12]}.json"
        tmp = self.dir / f"{name}.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / name)  # nunca expõe arquivo pela metade
        return f"s6://trajectories/{name}"


class S6BusService:
    def __init__(
        self,
        pipeline: S6SynthesisPipeline,
        store: TrajectoryStore,
        lexicon_version: str,
        markers: frozenset[str] = DEFAULT_MARKERS,
    ):
        self.pipeline = pipeline
        self.store = store
        self.lexicon_version = lexicon_version
        self.markers = markers
        self._lookup = {_norm(label): label for label in pipeline.lexicon}

    def handle_synthesis(self, payload: dict[str, Any], msg: dict[str, Any]) -> dict[str, Any]:
        glosses = payload.get("glosses")
        speed = payload.get("speed_factor")
        if not isinstance(glosses, list) or not glosses or not all(isinstance(g, str) for g in glosses):
            raise BusError("EMPTY_SEQUENCE", "glosses ausente ou vazia")
        if not isinstance(speed, (int, float)) or isinstance(speed, bool) or speed <= 0:
            raise BusError("INVALID_REQUEST", "speed_factor deve ser um número > 0")
        if payload.get("lexicon_version") != self.lexicon_version:
            raise BusError(
                "UNKNOWN_LEXICON_VERSION",
                f"léxico {payload.get('lexicon_version')!r} não carregado (disponível: {self.lexicon_version})",
            )

        # Glosa desconhecida não derruba o pedido: entra em unknown_signs e a
        # trajetória pula essa posição (contrato da spec).
        known: list[str] = []  # rótulos do léxico, na ordem
        orig_idx: list[int] = []  # posição original de cada um em `glosses`
        unknown: list[str] = []
        for i, gloss in enumerate(glosses):
            label = self._lookup.get(_norm(gloss))
            if label is None:
                unknown.append(gloss)
            else:
                known.append(label)
                orig_idx.append(i)
        if not known:
            raise BusError("EMPTY_SEQUENCE", f"nenhuma glosa encontrada no léxico: {unknown}")

        # speed_factor: o ritmo de reprodução sai do fps efetivo (frames
        # iguais, tempos em ms escalados).
        cfg = dataclasses.replace(self.pipeline.config, fps=self.pipeline.config.fps * float(speed))
        out = synthesize(known, self.pipeline.lexicon, cfg)

        sign_segments = [s for s in out.segments if s.tipo == "sinal"]
        segments, k = [], 0
        for s in out.segments:
            if s.tipo == "sinal":
                segments.append(
                    {"sign": glosses[orig_idx[k]], "start_ms": s.inicio_ms, "end_ms": s.fim_ms, "is_transition": False}
                )
                k += 1
            else:
                segments.append({"sign": None, "start_ms": s.inicio_ms, "end_ms": s.fim_ms, "is_transition": True})

        facial_markers = []
        for m in payload.get("markers") or []:
            try:
                name, lo, hi = m["marker"], int(m["from_gloss_index"]), int(m["to_gloss_index"])
            except (KeyError, TypeError, ValueError):
                raise BusError("INVALID_REQUEST", f"marcador malformado: {m!r}")
            if name not in self.markers:
                raise BusError("INVALID_REQUEST", f"marcador fora do conjunto fechado: {name}")
            covered = [k for k, o in enumerate(orig_idx) if lo <= o <= hi]
            if covered:
                facial_markers.append(
                    {
                        "marker": name,
                        "start_ms": sign_segments[covered[0]].inicio_ms,
                        "end_ms": sign_segments[covered[-1]].fim_ms,
                    }
                )

        frame_count = len(out.keypoints)
        ref = self.store.save(
            {
                "fps": cfg.fps,
                "frame_count": frame_count,
                "coordinates": "NORMALIZED",
                "frames": [to_spec_frame(f) for f in out.keypoints],
            }
        )
        return {
            "fps": cfg.fps,
            "frame_count": frame_count,
            "duration_ms": int(round(frame_count / cfg.fps * 1000)),
            "trajectory_ref": ref,
            "segments": segments,
            "facial_markers": facial_markers,
            "unknown_signs": unknown,
        }


def main():
    parser = argparse.ArgumentParser(description="S6 no barramento (síntese de Libras).")
    parser.add_argument("--lexicon-dir", default="lexicon")
    parser.add_argument("--lexicon-version", default="lexicon-a6-1.3")
    parser.add_argument("--trajectory-dir", default="trajectories")
    parser.add_argument("--group-id", default="S6-synthesis")
    parser.add_argument("--bootstrap", default=None, help="Sobrescreve ODS_KAFKA_BOOTSTRAP")
    parser.add_argument("--offset-reset", default="latest", choices=["latest", "earliest"])
    parser.add_argument("--transition-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pipeline = S6SynthesisPipeline(args.lexicon_dir, transition_frames=args.transition_frames, fps=args.fps)
    service = S6BusService(pipeline, TrajectoryStore(args.trajectory_dir), args.lexicon_version)
    CommandService(
        component=COMPONENT,
        group_id=args.group_id,
        handlers={COMMAND_SCHEMA: service.handle_synthesis},
        producer=BusProducer(args.bootstrap),
        bootstrap=args.bootstrap,
        auto_offset_reset=args.offset_reset,
    ).run()


if __name__ == "__main__":
    main()