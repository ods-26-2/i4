"""
tools/webcam_client.py
=======================
Captura a webcam, comprime cada quadro em JPEG e manda como mensagem
binária pro server.cpp em ws://<host>:18080/ws/video (um JPEG por mensagem).

Roda no Windows (onde está a webcam); o server roda no WSL.

    pip install opencv-python websockets
    python tools/webcam_client.py --url ws://localhost:18080/ws/video --preview

O server aceita UMA conexão por vez: feche este cliente (Ctrl+C) antes de
abrir outro, senão o novo é recusado.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
from websockets.sync.client import connect


def open_camera(index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    # CAP_DSHOW abre mais rápido no Windows; fora dele cai no backend padrão.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {index}")
    # MJPG costuma liberar 30 fps em webcams USB; em YUY2 muitas caem pra 10 fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    ap = argparse.ArgumentParser(description="Webcam -> JPEG -> WebSocket do server.cpp")
    ap.add_argument("--url", default="ws://localhost:18080/ws/video")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=25.0, help="taxa máxima de envio (o treino usou 25)")
    ap.add_argument("--quality", type=int, default=85, help="qualidade JPEG 1-100")
    ap.add_argument("--preview", action="store_true", help="mostra a imagem enviada (q fecha)")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"câmera {args.camera}: {w}x{h} (pedido: {args.width}x{args.height})")

    interval = 1.0 / args.fps
    params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    n = 0
    next_t = time.monotonic()
    print(f"conectando em {args.url} ...")
    try:
        with connect(args.url, max_size=None) as ws:
            print("conectado. Ctrl+C para encerrar.")
            while True:
                ok, bgr = cap.read()
                if not ok:
                    print("câmera parou de entregar frames")
                    break
                now = time.monotonic()
                if now < next_t:  # limita a taxa: o server processa em série
                    continue
                next_t = now + interval
                # NÃO espelhar: o modelo foi treinado com vídeos sem espelho.
                ok, jpg = cv2.imencode(".jpg", bgr, params)
                if not ok:
                    continue
                ws.send(jpg.tobytes())
                n += 1
                if args.preview:
                    cv2.imshow("enviando (q para sair)", bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    except ConnectionRefusedError:
        print("conexão recusada: o server.cpp está rodando? (porta 18080)")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"{n} frames enviados")


if __name__ == "__main__":
    main()