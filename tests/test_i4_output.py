import json

from pipeline.schema.i4_output import I4Output
from pipeline.schema.sign_annotation import SignAnnotation


def _sample_output() -> I4Output:
    return I4Output(
        sign=SignAnnotation(sinal="Aluno", confianca=0.9, inicio_ms=0, fim_ms=800, sinalizador_id="S1"),
        keypoints=[{"pose": [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 1.0}]}],
        marker_layout=[{"name": "pose", "offset": 0, "num_points": 1, "dims_per_point": 4, "point_dims": ("x", "y", "z", "visibility")}],
        frame_timestamps_ms=[0],
    )


def test_to_dict_bundles_sign_keypoints_and_marker_layout():
    output = _sample_output()
    data = output.to_dict()
    assert data["sign"]["sinal"] == "Aluno"
    assert data["keypoints"][0]["pose"][0]["x"] == 0.1
    assert data["marker_layout"][0]["name"] == "pose"
    assert data["frame_timestamps_ms"] == [0]


def test_to_json_roundtrip():
    output = _sample_output()
    parsed = json.loads(output.to_json())
    assert parsed["sign"]["confianca"] == 0.9
    assert parsed["keypoints"][0]["pose"][0]["visibility"] == 1.0
