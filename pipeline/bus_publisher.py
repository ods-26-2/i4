"""
pipeline/bus_publisher.py
==========================
I4 -> barramento. Publica, para cada sinal reconhecido:

    ods.i4.keypoints   um evento POR QUADRO (convenção 5 da spec)
    ods.i4.sign        um evento por sinal fechado

Consumidores (arquitetura 6c): S6 e A6. Perfil EV-SINAL.

Uso (ver também o trecho de integração no ao_vivo.py / pipeline.py):

    bus = I4BusPublisher(BusProducer(), source_id="cam0")
    bus.publish_sign(
        label=label, confidence=conf,
        keypoint_frames=sequence_to_keypoint_frames(seq),
        timestamps_ms=[...],            # epoch ms ABSOLUTO de cada frame
        image_size=(reader.w, reader.h),
    )
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional, Sequence

from ods_bus.client import BusProducer
from ods_bus.envelope import SequenceCounter, build_message, epoch_ms_to_iso
from ods_bus.keypoints import to_spec_frame

PRODUCER = "I4"


class I4BusPublisher:
    def __init__(
        self,
        producer: BusProducer,
        source_id: str = "cam0",
        artifact_version: str = "signs-minds-20-1.0",
        session_id: Optional[str] = None,
        gloss_fn: Callable[[str], str] = str.upper,
    ):
        self.producer = producer
        self.source_id = source_id
        self.artifact_version = artifact_version
        # P4 é quem cria o session_id na arquitetura; hoje a shm não o carrega,
        # então cada execução do I4 gera o seu. Trocar quando P4 publicar
        # ods.session.started.
        self.session_id = session_id or str(uuid.uuid4())
        self.gloss_fn = gloss_fn  # rótulo do modelo ("Aluno") -> glosa do léxico ("ALUNO")
        self._seq = SequenceCounter()
        self._frame_index = 0

    def _msg(self, schema: str, payload: dict, captured_at: str):
        return build_message(
            message_type="event",
            schema=schema,
            producer=PRODUCER,
            payload=payload,
            profile="EV-SINAL",
            captured_at=captured_at,
            source_id=self.source_id,
            session_id=self.session_id,
            sequence=self._seq.next(PRODUCER, self.session_id),
            artifact_version=self.artifact_version,
        )

    def publish_sign(
        self,
        *,
        label: str,
        confidence: float,
        keypoint_frames: Sequence[dict],
        timestamps_ms: Sequence[int],
        image_size: tuple[int, int],
        signer_id: Optional[str] = None,
        publish_keypoints: bool = True,
    ) -> None:
        if len(keypoint_frames) != len(timestamps_ms):
            raise ValueError("keypoint_frames e timestamps_ms precisam ter o mesmo tamanho")
        width, height = image_size

        if publish_keypoints:
            for frame, ts in zip(keypoint_frames, timestamps_ms):
                payload = {
                    "frame_index": self._frame_index,
                    "image_width": width,
                    "image_height": height,
                    "coordinates": "NORMALIZED",
                    **to_spec_frame(frame),
                }
                self.producer.publish(self._msg("ods.i4.keypoints", payload, epoch_ms_to_iso(ts)))
                self._frame_index += 1

        start_at, end_at = epoch_ms_to_iso(timestamps_ms[0]), epoch_ms_to_iso(timestamps_ms[-1])
        sign_payload = {
            "sign": self.gloss_fn(label),
            "confidence": round(float(confidence), 4),
            "start_at": start_at,
            "end_at": end_at,
        }
        if signer_id is not None:
            sign_payload["signer_id"] = signer_id
        # No evento de sinal, captured_at = start_at (spec).
        self.producer.publish(self._msg("ods.i4.sign", sign_payload, start_at))
        self.producer.flush(5.0)