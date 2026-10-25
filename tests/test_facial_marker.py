import json

from ods_bus.envelope import validate_message
from pipeline.bus_publisher import I4BusPublisher
from pipeline.schema.facial_marker import FacialMarker


class FakeProducer:
    def __init__(self):
        self.sent = []
        self.flushed = 0

    def publish(self, msg, key=None):
        self.sent.append(msg)

    def flush(self, timeout=10.0):
        self.flushed += 1


def _publisher():
    producer = FakeProducer()
    return I4BusPublisher(producer, source_id="cam0", session_id="sess-1"), producer


def test_publish_facial_marker_builds_valid_ev_sinal_message():
    bus, producer = _publisher()
    ok = bus.publish_facial_marker(FacialMarker("INTERROGATIVA_QU", 0.78123, 1_789_567_511_300, 1_789_567_512_100))
    assert ok is True
    (msg,) = producer.sent
    assert validate_message(msg, schema="ods.i4.facial_marker", profile="EV-SINAL") == (True, None)
    assert msg["producer"] == "I4"
    assert msg["payload"]["marker"] == "INTERROGATIVA_QU"
    assert msg["payload"]["confidence"] == 0.7812
    assert msg["payload"]["start_at"].endswith("Z") and msg["payload"]["end_at"].endswith("Z")
    assert msg["captured_at"] == msg["payload"]["start_at"]  # spec: captured_at = start_at
    assert msg["artifact_version"] == "facial-markers-0.1"
    json.dumps(msg)  # serializável


def test_neutral_is_not_published_by_default():
    bus, producer = _publisher()
    assert bus.publish_facial_marker(FacialMarker("NEUTRO", 0.9, 0, 500)) is False
    assert producer.sent == []
    assert bus.publish_facial_marker(FacialMarker("NEUTRO", 0.9, 0, 500), publish_neutral=True) is True


def test_marker_outside_a6_set_is_rejected():
    bus, producer = _publisher()
    try:
        bus.publish_facial_marker(FacialMarker("SURPRESA", 0.9, 0, 500), allowed_markers={"NEGACAO"})
    except ValueError:
        pass
    else:
        raise AssertionError("deveria rejeitar marcador que A6 não definiu")
    assert producer.sent == []


def test_invalid_interval_or_confidence_is_rejected():
    for bad in (FacialMarker("NEGACAO", 1.5, 0, 500), FacialMarker("NEGACAO", 0.5, 500, 500)):
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError("deveria rejeitar")


def test_sequence_numbers_increase_per_session():
    bus, producer = _publisher()
    bus.publish_facial_marker(FacialMarker("NEGACAO", 0.8, 0, 500))
    bus.publish_facial_marker(FacialMarker("AFIRMACAO", 0.8, 600, 1100))
    assert [m["sequence"] for m in producer.sent] == [0, 1]