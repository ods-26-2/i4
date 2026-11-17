"""ao_vivo.py: lê frames da shm, segmenta por pausa e roda o pipeline."""
import argparse, os, time
from pathlib import Path
import cv2
import numpy as np

from pipeline.live.leitor import FrameReader
from pipeline.feature_extraction.landmark_extractor import extract_landmarks
from pipeline.model.predict import SignPredictor
from pipeline.schema.sign_annotation import SignAnnotation

from pipeline.feature_extraction.expression_features import RollingBaseline, base_features
from pipeline.model.expression_classifier import ExpressionPredictor
from pipeline.model.expression_rules import RuleConfig, RuleExpressionPredictor
from pipeline.schema.facial_marker import FacialMarker

from ods_bus.client import BusProducer
from pipeline.bus_publisher import I4BusPublisher
from pipeline.feature_extraction.landmark_extractor import sequence_to_keypoint_frames

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pipeline.feature_extraction.landmark_extractor import (
    LandmarkExtractionConfig, ensure_model_downloaded, DEFAULT_MODEL_PATH,
    result_to_feature_vector, append_hand_velocity, sequence_to_keypoint_frames,
)
from pipeline.live.segmenter import SignSegmenter

SHM = "/dev/shm/frames_cam"
TARGET_FPS = 25.0        # o mesmo do PreprocessConfig (o treino usou isso)
SIZE = (224, 224)        # idem, resize esticado como no video_preprocessor


def abrir():
    while True:
        try:
            return FrameReader(SHM)
        except (FileNotFoundError, AssertionError):
            time.sleep(0.5)


def preparar(buf):
    """buf: [(bgr, ts_ms)] -> (frames RGB 224x224 a 25 fps, duração_ms).
    O servidor só grava 1 a cada 3 frames e só com rosto detectado, então os
    timestamps têm buracos. Reamostra por timestamp pra recuperar os 25 fps."""
    ts = np.array([t for _, t in buf], dtype=np.float64)
    dur = ts[-1] - ts[0]
    n = max(2, int(round(dur / 1000 * TARGET_FPS)))
    alvo = np.linspace(ts[0], ts[-1], n)
    idx = np.abs(ts[None, :] - alvo[:, None]).argmin(axis=1)

    cache = {}
    frames = []
    for i in idx:
        if i not in cache:
            rgb = cv2.cvtColor(buf[i][0], cv2.COLOR_BGR2RGB)
            cache[i] = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)
        frames.append(cache[i])
    return frames, int(dur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gap", type=float, default=1.0, help="s sem frame novo = fim do sinal")
    # 500ms descartava sinais curtos de verdade (ex.: "Não" balançando a
    # cabeça rápido) — o servidor já grava a ~10fps (1 a cada 3 de uma
    # câmera a 30fps), então o filtro de 4 frames abaixo já barra ruído
    # (toque acidental, mão passando no quadro); não precisa de um piso
    # de duração tão alto em cima disso.
    ap.add_argument("--min-ms", type=int, default=150, help="ignora trechos menores que isso")
    ap.add_argument("--max-s", type=float, default=6.0, help="corta trechos mais longos que isso")
    ap.add_argument("--expr-model", default=None, help="modelo treinado de expressões (.pt)")
    ap.add_argument("--expr-rules", default=None, help="expressões por regras: 'padrao' ou caminho do json de limiares")
    ap.add_argument("--publish", action="store_true", help="publica no barramento (sem isso, só imprime)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    predictor = SignPredictor(args.model, device=args.device)
    reader = abrir()
    inode = os.fstat(reader.f.fileno()).st_ino
    buf, last = [], time.time()
    print("Aguardando frames...", flush=True)
    

    if args.publish:
        from ods_bus.client import BusProducer, ensure_topics, topic_for

    if args.publish:
        ensure_topics([topic_for(s) for s in (
            "ods.i4.keypoints", "ods.i4.sign", "ods.i4.facial_marker"
        )])
    
    
    bus = None
    if args.publish:
        from ods_bus.client import ensure_topics, topic_for
        ensure_topics([topic_for(s) for s in ("ods.i4.keypoints", "ods.i4.sign", "ods.i4.facial_marker")])
        # artifact_version reflete o arquivo de checkpoint carregado (--model),
        # em vez do default hardcoded do publisher (modelo antigo de 20 classes)
        bus = I4BusPublisher(BusProducer(), source_id="cam0", artifact_version=Path(args.model).stem)
    
    # bus = I4BusPublisher(BusProducer(), source_id="cam0")
    
    if args.expr_model:
        expr = ExpressionPredictor(args.expr_model, device=args.device)
    elif args.expr_rules:
        expr = RuleExpressionPredictor(RuleConfig() if args.expr_rules == "padrao" else RuleConfig.load(args.expr_rules))
    else:
        expr = None
    rolling = RollingBaseline(seconds=10.0, fps=25.0)
    
    
    cfg = LandmarkExtractionConfig()
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model_downloaded(DEFAULT_MODEL_PATH))),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5, min_face_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5, min_hand_landmarks_confidence=0.5,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    seg = SignSegmenter(min_ms=args.min_ms, max_ms=int(args.max_s * 1000))
    rolling = RollingBaseline(seconds=10.0, fps=25.0)
        
    while True:
    # o servidor recria o arquivo quando o cliente reconecta
        try:
            if os.stat(SHM).st_ino != inode:
                reader.close()
                reader = abrir()
                inode = os.fstat(reader.f.fileno()).st_ino
                seg.reset()
        except FileNotFoundError:
            pass

        res = reader.ler()
        if res is None:
            time.sleep(0.005)
            continue

        bgr, ts = res
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), SIZE, interpolation=cv2.INTER_AREA)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        row = result_to_feature_vector(result, cfg)

        # a baseline de expressão precisa de frames em repouso: alimenta só quando não há sinal ativo
        if not seg.active:
            rolling.update(base_features(row[None, :])[0])

        estava_ativo = seg.active
        out = seg.feed(row, ts)

        if not estava_ativo and seg.active:
            print(">> Sinal detectado, gravando...", flush=True)

        if estava_ativo and not seg.active and out is None:
            print(">> Trecho descartado (curto demais)", flush=True)

        if out is None:
            continue

        rows, tss = out
        t0 = time.time()
        
        print(f">> Processando ({tss[-1] - tss[0]} ms, {len(tss)} frames)...", flush=True)

        alvo = np.linspace(tss[0], tss[-1], max(2, int((tss[-1] - tss[0]) / 1000 * TARGET_FPS)))
        idx = np.abs(np.array(tss)[None, :] - alvo[:, None]).argmin(axis=1)
        seq = append_hand_velocity(np.stack([rows[i] for i in idx]), cfg)
        ts_list = np.linspace(tss[0], tss[-1], len(seq)).astype(int).tolist()

        label, conf = predictor.predict(seq)
        print(f">> Pronto em {time.time() - t0:.2f} s", flush=True)
        print(SignAnnotation(sinal=label, confianca=conf,
                             inicio_ms=int(tss[0]), fim_ms=int(tss[-1])).to_json(indent=2), flush=True)


        if args.publish:
            bus.publish_sign(label=label, confidence=conf,
                             keypoint_frames=sequence_to_keypoint_frames(seq),
                             timestamps_ms=ts_list, image_size=(reader.w, reader.h),
                             publish_keypoints=False)

        if expr:
            baseline = rolling.value
            if np.any(baseline):
                segs = expr.predict_segments(seq, ts_list, baseline=baseline)
                for s in segs:
                    print(f"  expressão: {s.marker} (conf {s.confidence:.2f})", flush=True)
                    if args.publish:
                        bus.publish_facial_marker(FacialMarker(s.marker, s.confidence, s.start_ms, s.end_ms))
                    else:
                        print("  expressão: aquecendo linha de base", flush=True)
                    for b in base_features(seq):
                        rolling.update(b)
            else:
                print(f"trecho curto demais ({len(buf)} frames), ignorado")
            buf = []
        else:
            time.sleep(0.005)


if __name__ == "__main__":
    main()