"""
landmark_extractor.py
======================
Etapa [3] do pipeline: roda o MediaPipe Holistic Landmarker frame a
frame e gera uma sequência de vetores de features por vídeo
(coordenadas normalizadas de pose corporal, mãos e rosto).

Usa a Tasks API do MediaPipe (`mediapipe.tasks.python.vision`), que é a
API disponível no mediapipe>=0.10 — a antiga `mp.solutions.holistic`
não existe mais nas versões atuais do pacote. Essa API precisa de um
modelo `.task` local; `ensure_model_downloaded` baixa o modelo oficial
do Google (holistic_landmarker, float16) na primeira execução e o
mantém em cache em `feature_extraction/models/`.

Contrato de entrada/saída:
    extract_landmarks(frames, config) -> np.ndarray[T, feature_dim]

Cada linha do array é o vetor de landmarks de um frame; componentes não
detectados (pose/mão/rosto ausente no frame) ficam preenchidos com
zeros nessa parte do vetor, preservando o comprimento T da sequência.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "holistic_landmarker.task"

# Contagens de landmarks do modelo Holistic Landmarker do MediaPipe.
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS_FULL = 478

POSE_DIM = NUM_POSE_LANDMARKS * 4  # x, y, z, visibility
HAND_DIM = NUM_HAND_LANDMARKS * 3  # x, y, z
FACE_DIM_FULL = NUM_FACE_LANDMARKS_FULL * 3  # x, y, z

# Subconjunto semântico da malha facial (sobrancelhas + olhos + boca), a
# união dos índices de FaceLandmarksConnections.FACE_LANDMARKS_{LIPS,
# LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW} do próprio MediaPipe.
# Motivo pra não usar os 478 pontos da malha densa: em Libras a face só
# carrega informação linguística nos marcadores não-manuais (sobrancelha
# levantada em pergunta, boca acompanhando o sinal, olhos semicerrados
# etc.) — o resto da malha (bochecha, testa, contorno do rosto) é forma/
# identidade do rosto, não expressão, e pipelines de reconhecimento de
# expressão facial (ex: os 68 pontos clássicos do dlib) também usam só
# essas regiões. Manter os 478 pontos infla o vetor de features com
# ruído de alta dimensão (~85% do vetor total) sem ganho de sinal, o que
# é especialmente ruim treinando com poucos vídeos por classe.
FACE_SEMANTIC_LANDMARK_INDICES: tuple[int, ...] = (
    0, 7, 13, 14, 17, 33, 37, 39, 40, 46, 52, 53, 55, 61, 63, 65, 66, 70,
    78, 80, 81, 82, 84, 87, 88, 91, 95, 105, 107, 133, 144, 145, 146, 153,
    154, 155, 157, 158, 159, 160, 161, 163, 173, 178, 181, 185, 191, 246,
    249, 263, 267, 269, 270, 276, 282, 283, 285, 291, 293, 295, 296, 300,
    308, 310, 311, 312, 314, 317, 318, 321, 324, 334, 336, 362, 373, 374,
    375, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 402, 405, 409,
    415, 466,
)
NUM_FACE_LANDMARKS_REDUCED = len(FACE_SEMANTIC_LANDMARK_INDICES)
FACE_DIM_REDUCED = NUM_FACE_LANDMARKS_REDUCED * 3  # x, y, z

# Índices dos ombros no modelo de pose do MediaPipe (iguais entre a API
# legada e a Tasks API — `vision.PoseLandmark.LEFT_SHOULDER`/`RIGHT_SHOULDER`).
# Usados como referência pra normalização corporal dos landmarks.
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
# Índice do pulso no modelo de mão do MediaPipe (HandLandmark.WRIST) —
# usado como referência pra normalização local da mão (ver
# include_hand_local_norm abaixo).
WRIST_IDX = 0


@dataclass
class LandmarkExtractionConfig:
    model_path: str | Path = DEFAULT_MODEL_PATH
    include_face: bool = True
    # "reduced" (padrão): só sobrancelhas/olhos/boca (92 pontos, ver
    # FACE_SEMANTIC_LANDMARK_INDICES acima). "full": os 478 pontos da
    # malha densa do MediaPipe.
    face_mode: str = "reduced"
    min_detection_confidence: float = 0.5
    min_landmarks_confidence: float = 0.5
    # Centraliza todos os landmarks (pose/mãos/rosto) no ponto médio dos
    # ombros e escala pela largura dos ombros, antes de montar o vetor de
    # features. Sem isso, as coordenadas são absolutas em tela — o modelo
    # aprende a distância/posição do sinalizador em relação à câmera e o
    # porte físico dele, em vez do movimento do sinal, o que generaliza mal
    # pra sinalizadores novos (ver leave-one-signer-out em `model/train.py`).
    normalize_landmarks: bool = True
    # Além da normalização global (corpo inteiro, escala = largura dos
    # ombros), adiciona uma segunda cópia de cada mão normalizada pelo
    # próprio pulso e pelo próprio tamanho (maior distância pulso->ponta
    # de dedo). Na normalização global, a mão ocupa só uma fração pequena
    # da escala (largura dos ombros), então variações finas de configuração
    # dos dedos (classificador de sinal, não de posição) ficam comprimidas
    # perto da resolução de ruído do landmark. A cópia local amplia esse
    # sinal. Ver confusões de handshape no relatório (Ruim<->Maca,
    # Vacina<->Aluno) — motivação direta pra essa feature.
    include_hand_local_norm: bool = True
    # Adiciona, por frame, a velocidade (delta pro frame anterior) dos
    # landmarks de mão já normalizados globalmente. Sinais em Libras
    # carregam muito da informação na trajetória/movimento da mão, não só
    # na forma estática por frame — isso dá esse sinal explicitamente ao
    # LSTM em vez de depender só da recorrência pra inferir. Calculado em
    # `extract_landmarks` (precisa do frame anterior), não aqui.
    include_hand_velocity: bool = True

    @property
    def face_dim(self) -> int:
        if not self.include_face:
            return 0
        return FACE_DIM_REDUCED if self.face_mode == "reduced" else FACE_DIM_FULL

    @property
    def hand_local_dim(self) -> int:
        return 2 * HAND_DIM if self.include_hand_local_norm else 0

    @property
    def hand_velocity_dim(self) -> int:
        return 2 * HAND_DIM if self.include_hand_velocity else 0

    @property
    def static_feature_dim(self) -> int:
        """Dimensão do vetor por frame antes de anexar a velocidade (que
        depende do frame anterior e é anexada em `extract_landmarks`)."""
        return POSE_DIM + 2 * HAND_DIM + self.face_dim + self.hand_local_dim

    @property
    def feature_dim(self) -> int:
        return self.static_feature_dim + self.hand_velocity_dim

    @property
    def left_hand_global_offset(self) -> int:
        return POSE_DIM

    @property
    def right_hand_global_offset(self) -> int:
        return POSE_DIM + HAND_DIM


def ensure_model_downloaded(model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Garante que o modelo `.task` do Holistic Landmarker está em disco,
    baixando-o do repositório oficial do MediaPipe se necessário."""
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def _flatten(landmarks: Sequence[Any] | None, num_points: int, include_visibility: bool) -> np.ndarray:
    """Achata uma lista de landmarks pra um vetor de tamanho fixo
    (num_points * (4 se include_visibility else 3)), preenchendo com
    zeros pontos ausentes (landmarks=None/vazio ou lista mais curta)."""
    per_point = 4 if include_visibility else 3
    vec = np.zeros(num_points * per_point, dtype=np.float32)
    for i, lm in enumerate(landmarks or []):
        if i >= num_points:
            break
        base = i * per_point
        vec[base] = lm.x or 0.0
        vec[base + 1] = lm.y or 0.0
        vec[base + 2] = lm.z or 0.0
        if include_visibility:
            vec[base + 3] = lm.visibility or 0.0
    return vec


def _flatten_by_indices(
    landmarks: Sequence[Any] | None, indices: Sequence[int], include_visibility: bool
) -> np.ndarray:
    """Como `_flatten`, mas em vez dos primeiros `num_points` pontos na
    ordem em que vêm, extrai só os índices específicos passados (usado
    pro subconjunto semântico de landmarks faciais)."""
    per_point = 4 if include_visibility else 3
    vec = np.zeros(len(indices) * per_point, dtype=np.float32)
    if not landmarks:
        return vec
    n_available = len(landmarks)
    for out_i, idx in enumerate(indices):
        if idx >= n_available:
            continue
        lm = landmarks[idx]
        base = out_i * per_point
        vec[base] = lm.x or 0.0
        vec[base + 1] = lm.y or 0.0
        vec[base + 2] = lm.z or 0.0
        if include_visibility:
            vec[base + 3] = lm.visibility or 0.0
    return vec


def _body_center_scale(pose_landmarks: Sequence[Any] | None) -> tuple[tuple[float, float, float], float]:
    """Centro (ponto médio dos ombros) e escala (largura dos ombros) pra
    normalizar os landmarks num referencial centrado no corpo do
    sinalizador. Sem ombros detectados, usa um referencial neutro
    (centro da imagem, escala 1) — os landmarks ficam como vieram."""
    if not pose_landmarks or len(pose_landmarks) <= RIGHT_SHOULDER_IDX:
        return (0.5, 0.5, 0.0), 1.0

    left = pose_landmarks[LEFT_SHOULDER_IDX]
    right = pose_landmarks[RIGHT_SHOULDER_IDX]
    center = ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0, (left.z + right.z) / 2.0)
    scale = float(np.hypot(left.x - right.x, left.y - right.y))
    if scale < 1e-6:
        scale = 1.0
    return center, scale


def _normalize_points(
    landmarks: Sequence[Any] | None, center: tuple[float, float, float], scale: float
) -> Sequence[Any] | None:
    """Reexpressa cada landmark em `(x,y,z)` relativo a `center`, escalado
    por `scale`; preserva `visibility` quando presente. Lista vazia/None
    passa direto (`_flatten`/`_flatten_by_indices` já tratam isso como
    "nada detectado" e preenchem zeros)."""
    if not landmarks:
        return landmarks
    cx, cy, cz = center
    return [
        SimpleNamespace(
            x=(lm.x - cx) / scale,
            y=(lm.y - cy) / scale,
            z=(lm.z - cz) / scale,
            visibility=getattr(lm, "visibility", None),
        )
        for lm in landmarks
    ]


def _hand_local_normalize(hand_landmarks: Sequence[Any] | None) -> Sequence[Any] | None:
    """Re-expressa uma mão em referencial local: centrada no próprio pulso
    (WRIST_IDX) e escalada pela maior distância pulso->ponta de dedo dessa
    mesma mão. Ao contrário de `_normalize_points` (referencial do corpo
    inteiro), isso amplia a configuração dos dedos pra uma escala fixa
    independente de quão perto/longe a mão está do corpo — é o sinal de
    "formato da mão" (handshape) isolado da posição da mão no sinal."""
    if not hand_landmarks or len(hand_landmarks) <= WRIST_IDX:
        return hand_landmarks

    wrist = hand_landmarks[WRIST_IDX]
    cx, cy, cz = wrist.x, wrist.y, wrist.z
    scale = max(
        (np.hypot(lm.x - cx, lm.y - cy) for lm in hand_landmarks),
        default=0.0,
    )
    if scale < 1e-6:
        scale = 1.0
    return [
        SimpleNamespace(
            x=(lm.x - cx) / scale,
            y=(lm.y - cy) / scale,
            z=(lm.z - cz) / scale,
            visibility=getattr(lm, "visibility", None),
        )
        for lm in hand_landmarks
    ]


def result_to_feature_vector(result: Any, config: LandmarkExtractionConfig) -> np.ndarray:
    """Converte um `HolisticLandmarkerResult` (ou qualquer objeto com os
    mesmos atributos) num único vetor de features pro frame (sem a
    velocidade de mão, que depende do frame anterior — ver
    `extract_landmarks`/`append_hand_velocity`).

    Ordem das partes no vetor (ver também `describe_marker_layout`):
    pose, left_hand (global), right_hand (global), [face], [left_hand
    (local), right_hand (local)]."""
    pose_landmarks = result.pose_landmarks
    left_hand_landmarks = result.left_hand_landmarks
    right_hand_landmarks = result.right_hand_landmarks
    face_landmarks = result.face_landmarks

    if config.normalize_landmarks:
        center, scale = _body_center_scale(pose_landmarks)
        pose_landmarks = _normalize_points(pose_landmarks, center, scale)
        left_hand_landmarks = _normalize_points(left_hand_landmarks, center, scale)
        right_hand_landmarks = _normalize_points(right_hand_landmarks, center, scale)
        face_landmarks = _normalize_points(face_landmarks, center, scale)

    parts = [
        _flatten(pose_landmarks, NUM_POSE_LANDMARKS, include_visibility=True),
        _flatten(left_hand_landmarks, NUM_HAND_LANDMARKS, include_visibility=False),
        _flatten(right_hand_landmarks, NUM_HAND_LANDMARKS, include_visibility=False),
    ]
    if config.include_face:
        if config.face_mode == "reduced":
            parts.append(
                _flatten_by_indices(face_landmarks, FACE_SEMANTIC_LANDMARK_INDICES, include_visibility=False)
            )
        else:
            parts.append(_flatten(face_landmarks, NUM_FACE_LANDMARKS_FULL, include_visibility=False))
    if config.include_hand_local_norm:
        # Normalização local usa as mãos ORIGINAIS (antes do centro/escala
        # do corpo), já que o referencial é o próprio pulso da mão.
        parts.append(_flatten(_hand_local_normalize(result.left_hand_landmarks), NUM_HAND_LANDMARKS, include_visibility=False))
        parts.append(_flatten(_hand_local_normalize(result.right_hand_landmarks), NUM_HAND_LANDMARKS, include_visibility=False))
    return np.concatenate(parts)


def append_hand_velocity(sequence: np.ndarray, config: LandmarkExtractionConfig) -> np.ndarray:
    """Anexa, a cada frame, o delta (velocidade) das mãos (referencial
    global) em relação ao frame anterior; o primeiro frame recebe
    velocidade zero (não há frame anterior). Opera sobre a sequência já
    montada por `result_to_feature_vector` (shape (T, static_feature_dim)),
    já que a velocidade é a única parte do vetor que depende de mais de um
    frame. No-op se `config.include_hand_velocity` for False."""
    if not config.include_hand_velocity:
        return sequence
    if sequence.shape[0] == 0:
        return sequence

    lo = config.left_hand_global_offset
    ro = config.right_hand_global_offset
    hands = np.concatenate(
        [sequence[:, lo : lo + HAND_DIM], sequence[:, ro : ro + HAND_DIM]], axis=1
    )
    velocity = np.zeros_like(hands)
    velocity[1:] = hands[1:] - hands[:-1]
    return np.concatenate([sequence, velocity.astype(sequence.dtype)], axis=1)


def describe_marker_layout(config: LandmarkExtractionConfig | None = None) -> list[dict[str, Any]]:
    """Definição estável dos marcadores que compõem o vetor de features —
    o contrato que S6/A6 usam pra interpretar `ods.i4.keypoints` sem
    depender dos offsets internos deste módulo (requisito "Definição dos
    marcadores de expressão e mãos" da arquitetura). Cada bloco descreve um
    grupo contíguo de landmarks: nome, offset/largura no vetor achatado,
    quantos pontos e quantas dimensões por ponto."""
    config = config or LandmarkExtractionConfig()
    blocks: list[dict[str, Any]] = [
        {"name": "pose", "offset": 0, "num_points": NUM_POSE_LANDMARKS, "dims_per_point": 4, "point_dims": ("x", "y", "z", "visibility")},
        {"name": "left_hand_global", "offset": config.left_hand_global_offset, "num_points": NUM_HAND_LANDMARKS, "dims_per_point": 3, "point_dims": ("x", "y", "z")},
        {"name": "right_hand_global", "offset": config.right_hand_global_offset, "num_points": NUM_HAND_LANDMARKS, "dims_per_point": 3, "point_dims": ("x", "y", "z")},
    ]
    offset = POSE_DIM + 2 * HAND_DIM
    if config.include_face:
        num_face = NUM_FACE_LANDMARKS_REDUCED if config.face_mode == "reduced" else NUM_FACE_LANDMARKS_FULL
        blocks.append({"name": "face", "offset": offset, "num_points": num_face, "dims_per_point": 3, "point_dims": ("x", "y", "z")})
        offset += config.face_dim
    if config.include_hand_local_norm:
        blocks.append({"name": "left_hand_local", "offset": offset, "num_points": NUM_HAND_LANDMARKS, "dims_per_point": 3, "point_dims": ("x", "y", "z")})
        offset += HAND_DIM
        blocks.append({"name": "right_hand_local", "offset": offset, "num_points": NUM_HAND_LANDMARKS, "dims_per_point": 3, "point_dims": ("x", "y", "z")})
        offset += HAND_DIM
    if config.include_hand_velocity:
        blocks.append({"name": "left_hand_velocity", "offset": offset, "num_points": NUM_HAND_LANDMARKS, "dims_per_point": 3, "point_dims": ("vx", "vy", "vz")})
        offset += HAND_DIM
        blocks.append({"name": "right_hand_velocity", "offset": offset, "num_points": NUM_HAND_LANDMARKS, "dims_per_point": 3, "point_dims": ("vx", "vy", "vz")})
        offset += HAND_DIM
    assert offset == config.feature_dim, "describe_marker_layout saiu de sincronia com feature_dim"
    return blocks


def sequence_to_keypoint_frames(
    sequence: np.ndarray, config: LandmarkExtractionConfig | None = None
) -> list[dict[str, list[dict[str, float]]]]:
    """Reestrutura a sequência achatada (T, feature_dim) de volta em
    keypoints nomeados por frame — os "keypoints e marcadores brutos,
    direto" que I4 publica no barramento pra S6/A6, usando
    `describe_marker_layout` como contrato de offsets. Cada frame vira
    `{"pose": [{"x":..,"y":..,"z":..,"visibility":..}, ...], "left_hand_global": [...], ...}`."""
    config = config or LandmarkExtractionConfig()
    layout = describe_marker_layout(config)
    frames: list[dict[str, list[dict[str, float]]]] = []
    for row in sequence:
        frame: dict[str, list[dict[str, float]]] = {}
        for block in layout:
            start = block["offset"]
            width = block["num_points"] * block["dims_per_point"]
            values = row[start : start + width].reshape(block["num_points"], block["dims_per_point"])
            frame[block["name"]] = [dict(zip(block["point_dims"], point.tolist())) for point in values]
        frames.append(frame)
    return frames


def extract_landmarks(
    frames: list[np.ndarray], config: LandmarkExtractionConfig | None = None
) -> np.ndarray:
    """Contrato público da etapa [3]: frames RGB -> sequência de landmarks."""
    config = config or LandmarkExtractionConfig()
    model_path = ensure_model_downloaded(config.model_path)

    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_face_detection_confidence=config.min_detection_confidence,
        min_pose_landmarks_confidence=config.min_landmarks_confidence,
        min_hand_landmarks_confidence=config.min_landmarks_confidence,
    )

    sequence = np.zeros((len(frames), config.static_feature_dim), dtype=np.float32)

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = landmarker.detect(mp_image)
            sequence[i] = result_to_feature_vector(result, config)

    return append_hand_velocity(sequence, config)


def save_landmark_sequence(
    video_id: str,
    class_label: str,
    signer_id: str,
    sequence: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Persiste a sequência de landmarks de um vídeo em parquet, associada
    ao `video_id` e sem perder o rótulo (`class`) nem o sinalizador."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(sequence, columns=[f"feat_{i}" for i in range(sequence.shape[1])])
    df.insert(0, "frame_idx", np.arange(sequence.shape[0]))
    df.insert(0, "signer_id", signer_id)
    df.insert(0, "class", class_label)
    df.insert(0, "video_id", video_id)

    safe_name = Path(video_id).stem
    out_path = output_dir / f"{safe_name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def load_landmark_sequence(path: str | Path) -> tuple[np.ndarray, str, str, str]:
    """Lê de volta (sequence, video_id, class_label, signer_id) de um
    parquet gerado por `save_landmark_sequence`."""
    df = pd.read_parquet(path)
    df = df.sort_values("frame_idx")

    video_id = str(df["video_id"].iloc[0])
    class_label = str(df["class"].iloc[0])
    signer_id = str(df["signer_id"].iloc[0])

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    sequence = df[feature_cols].to_numpy(dtype=np.float32)
    return sequence, video_id, class_label, signer_id
