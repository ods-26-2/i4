"""
ods_bus/keypoints.py
=====================
Converte um frame no formato interno do I4 (`sequence_to_keypoint_frames`:
dict de blocos nomeados, com mãos locais e velocidade) para o formato do
contrato `ods.i4.keypoints`, que é o mesmo de cada quadro do artefato
KEYPOINT_SEQUENCE:

    {"body": [[x,y,z,visibility] x33] ou [],
     "left_hand": [[x,y,z] x21] ou None,
     "right_hand": [[x,y,z] x21] ou None,
     "face": [[x,y,z] x92] ou None}

O extrator preenche com ZEROS o que não detecta; aqui isso vira [] (corpo)
ou None (mãos/rosto), como o contrato pede.
"""

from __future__ import annotations

from typing import Any, Optional

DECIMALS = 4  # reduz o tamanho da mensagem; 1e-4 é abaixo do ruído do landmark


def _all_zero(points: Optional[list[dict[str, float]]]) -> bool:
    return not points or all(v == 0 for p in points for v in p.values())


def _round(v: float) -> float:
    return round(float(v), DECIMALS)


def _xyz(points: Optional[list[dict[str, float]]]) -> Optional[list[list[float]]]:
    if _all_zero(points):
        return None
    return [[_round(p["x"]), _round(p["y"]), _round(p["z"])] for p in points]


def to_spec_frame(frame: dict[str, Any]) -> dict[str, Any]:
    pose = frame.get("pose")
    body = (
        []
        if _all_zero(pose)
        else [[_round(p["x"]), _round(p["y"]), _round(p["z"]), _round(p.get("visibility", 0.0))] for p in pose]
    )
    return {
        "body": body,
        "left_hand": _xyz(frame.get("left_hand_global")),
        "right_hand": _xyz(frame.get("right_hand_global")),
        "face": _xyz(frame.get("face")),
    }