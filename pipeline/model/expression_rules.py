"""
expression_rules.py
====================
Detector de marcadores faciais por REGRAS, sem treino: usa as mesmas
features de `expression_features.py`, mas decide com limiares em vez de um
modelo aprendido. Existe porque ainda não há vídeos rotulados com os
marcadores de A6 pra treinar `expression_classifier.py`. Quando houver, o
modelo treinado entra no lugar com a mesma interface (`predict_segments`).

As regras seguem a descrição de A6 (lexicon-a6-1.3):

    NEGACAO           balanço de cabeça    -> yaw oscila (vai e volta)
    AFIRMACAO         aceno de cabeça      -> pitch oscila
    INTERROGATIVA_SN  sobrancelhas elevadas -> brow_altura acima do repouso
    INTERROGATIVA_QU  sobrancelhas franzidas (e cabeça à frente)
                                           -> brow_altura abaixo + brow_juntas menor
    NEUTRO            sem marcador

Tudo é relativo à linha de base da pessoa (as features já chegam com ela
subtraída), então os limiares valem pra qualquer sinalizador.

IMPORTANTE: os limiares padrão de `RuleConfig` são PONTO DE PARTIDA, não
valores medidos. Ajuste com `calibrate` (mede em dados que já temos quantas
janelas disparam por engano) e teste ao vivo fazendo cada expressão. Salvar
e carregar a calibração: `RuleConfig.save/load` (json).

Confiança: a pontuação de cada regra vale 0,5 exatamente no limiar e sobe
até 1,0 em 2x o limiar; por isso o `min_confidence=0.5` de
`segments_from_probs` equivale a "passou do limiar".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from pipeline.feature_extraction.expression_features import (
    BASE_FEATURE_NAMES,
    base_features,
    compute_baseline,
    sequence_features,
)
from pipeline.model.expression_windows import Segment, segments_from_probs, sliding_windows

# Mesma ordem (alfabética) que o classificador treinado usa em `class_names`.
CLASS_NAMES = ("AFIRMACAO", "INTERROGATIVA_QU", "INTERROGATIVA_SN", "NEGACAO", "NEUTRO")
_AFF, _QU, _SN, _NEG, _NEU = range(5)

_I = {name: i for i, name in enumerate(BASE_FEATURE_NAMES)}


@dataclass
class RuleConfig:
    win: int = 20                # frames por janela (~0,8 s a 25 fps)
    stride: int = 5
    fps: float = 25.0
    min_frames: int = 12         # sequência mais curta que isso não é avaliada
    # --- movimento de cabeça (amplitude = percentil 90 - percentil 10 na janela)
    shake_amp: float = 0.12      # yaw: amplitude mínima do balanço (NEGACAO)
    nod_amp: float = 0.05        # pitch: amplitude mínima do aceno (AFIRMACAO)
    min_reversals: int = 2       # idas e vindas mínimas na janela
    reversal_frac: float = 0.4   # uma volta só conta se andar >= 40% da amplitude
    # --- sobrancelha (mediana da janela, em relação ao repouso da pessoa)
    brow_up: float = 0.035       # brow_altura acima de +x    -> INTERROGATIVA_SN
    brow_down: float = 0.025     # brow_altura abaixo de -x   -> INTERROGATIVA_QU
    furrow: float = 0.015        # brow_juntas abaixo de -x   -> INTERROGATIVA_QU

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RuleConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------ medidas

def _score(value: float, threshold: float) -> float:
    """0,5 no limiar, 1,0 em 2x o limiar, 0 se zero ou negativo."""
    if threshold <= 0:
        return 0.0
    return float(np.clip(value / (2.0 * threshold), 0.0, 1.0))


def oscillation(x: np.ndarray, reversal_frac: float = 0.4) -> tuple[float, int]:
    """(amplitude, idas e vindas) de um sinal 1-D na janela.
    Amplitude = percentil 90 - percentil 10 (ignora um pico isolado).
    Uma volta só conta quando o sinal anda >= reversal_frac * amplitude
    desde o último extremo, o que descarta tremor pequeno."""
    if len(x) < 4:
        return 0.0, 0
    k = 3
    xs = np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    amp = float(np.percentile(xs, 90) - np.percentile(xs, 10))
    if amp <= 1e-9:
        return 0.0, 0
    min_move = reversal_frac * amp
    reversals, direction, extreme = 0, 0, xs[0]
    for v in xs[1:]:
        if direction == 0:
            if v - extreme >= min_move:
                direction, extreme = 1, v
            elif extreme - v >= min_move:
                direction, extreme = -1, v
        elif direction == 1:
            if v > extreme:
                extreme = v
            elif extreme - v >= min_move:
                direction, extreme, reversals = -1, v, reversals + 1
        else:
            if v < extreme:
                extreme = v
            elif v - extreme >= min_move:
                direction, extreme, reversals = 1, v, reversals + 1
    return amp, reversals


def window_scores(window: np.ndarray, cfg: RuleConfig) -> np.ndarray:
    """(win, N_FEATURES) -> pontuação 0..1 de cada classe (sem NEUTRO)."""
    yaw_amp, yaw_rev = oscillation(window[:, _I["yaw"]], cfg.reversal_frac)
    pit_amp, pit_rev = oscillation(window[:, _I["pitch"]], cfg.reversal_frac)

    rev_factor = lambda r: min(1.0, r / max(cfg.min_reversals, 1))
    neg = _score(yaw_amp, cfg.shake_amp) * rev_factor(yaw_rev)
    aff = _score(pit_amp, cfg.nod_amp) * rev_factor(pit_rev)
    # O eixo que mais passou do próprio limiar ganha; o outro é reduzido
    # (balançar a cabeça sempre mexe um pouco no pitch, e vice-versa).
    ratio_yaw, ratio_pit = yaw_amp / max(cfg.shake_amp, 1e-6), pit_amp / max(cfg.nod_amp, 1e-6)
    if ratio_yaw >= ratio_pit:
        aff *= 0.3
    else:
        neg *= 0.3

    brow = float(np.median(window[:, _I["brow_altura"]]))
    gap = float(np.median(window[:, _I["brow_juntas"]]))
    sn = _score(brow, cfg.brow_up)
    qu = min(_score(-brow, cfg.brow_down), _score(-gap, cfg.furrow))

    out = np.zeros(4, dtype=np.float64)
    out[_AFF], out[_QU], out[_SN], out[_NEG] = aff, qu, sn, neg
    return out


def scores_to_probs(scores: np.ndarray) -> np.ndarray:
    """(N, 4) pontuações -> (N, 5) 'probabilidades' na ordem de CLASS_NAMES:
    o vencedor fica com a própria pontuação e NEUTRO com o resto."""
    probs = np.zeros((len(scores), len(CLASS_NAMES)), dtype=np.float64)
    for i, s in enumerate(scores):
        w = int(np.argmax(s))
        probs[i, w] = s[w]
        probs[i, _NEU] = 1.0 - s[w]
    return probs


# ---------------------------------------------------------------- predictor

class RuleExpressionPredictor:
    """Mesma interface de `ExpressionPredictor` (modelo treinado), mas por regras."""

    def __init__(self, config: Optional[RuleConfig] = None):
        self.config = config or RuleConfig()
        self.class_names = list(CLASS_NAMES)

    def window_probs(self, sequence: np.ndarray, baseline: Optional[np.ndarray] = None):
        cfg = self.config
        feats = sequence_features(sequence, baseline=baseline, fps=cfg.fps)
        if len(feats) < cfg.min_frames:
            return np.empty((0, len(CLASS_NAMES))), [], cfg.win
        if len(feats) < cfg.win:  # sinal curto: avalia a sequência inteira como uma janela
            windows, starts = feats[None, :, :], [0]
            win = len(feats)
        else:
            windows, starts = sliding_windows(feats, cfg.win, cfg.stride)
            win = cfg.win
        scores = np.stack([window_scores(w, cfg) for w in windows])
        return scores_to_probs(scores), starts, win

    def predict_segments(
        self,
        sequence: np.ndarray,
        timestamps_ms: Sequence[int],
        baseline: Optional[np.ndarray] = None,
        **segment_kwargs,
    ) -> list[Segment]:
        probs, starts, win = self.window_probs(sequence, baseline)
        segment_kwargs.setdefault("min_confidence", 0.5)
        return segments_from_probs(probs, starts, win, timestamps_ms, self.class_names, **segment_kwargs)


# --------------------------------------------------------------- calibração

def _window_measures(window: np.ndarray, cfg: RuleConfig) -> dict[str, float]:
    yaw_amp, _ = oscillation(window[:, _I["yaw"]], cfg.reversal_frac)
    pit_amp, _ = oscillation(window[:, _I["pitch"]], cfg.reversal_frac)
    return {
        "yaw_amp": yaw_amp,
        "pitch_amp": pit_amp,
        "brow_altura": float(np.median(window[:, _I["brow_altura"]])),
        "brow_juntas": float(np.median(window[:, _I["brow_juntas"]])),
    }


def calibrate(sequences: Sequence[np.ndarray], cfg: Optional[RuleConfig] = None, percentile: float = 99.0):
    """Mede, em sequências SEM o marcador (ex.: sinais isolados de rosto
    neutro), o que as janelas produzem, e sugere limiares acima do
    `percentile`. Devolve (config_sugerida, taxa_de_disparo_com_a_config_atual).
    O limiar só sobe em relação ao atual; confirme ao vivo que expressões
    discretas ainda passam."""
    cfg = cfg or RuleConfig()
    meas: dict[str, list[float]] = {k: [] for k in ("yaw_amp", "pitch_amp", "brow_altura", "brow_juntas")}
    fired = np.zeros(len(CLASS_NAMES))
    total = 0
    for seq in sequences:
        feats = sequence_features(seq, fps=cfg.fps)
        windows, _ = sliding_windows(feats, cfg.win, cfg.stride)
        for w in windows:
            for k, v in _window_measures(w, cfg).items():
                meas[k].append(v)
            probs = scores_to_probs(window_scores(w, cfg)[None, :])[0]
            if probs.argmax() != _NEU and probs.max() >= 0.5:
                fired[probs.argmax()] += 1
            total += 1
    if total == 0:
        raise ValueError("nenhuma janela avaliada (sequências curtas demais)")
    p = lambda k, q: float(np.percentile(meas[k], q))
    margin = 1.1  # folga: o que está exatamente no percentil não deve disparar
    # Só SOBE o limiar (nunca abaixo do padrão): dados sem expressão dizem
    # onde o detector dispara por engano, mas não dizem quão discreta pode
    # ser uma expressão de verdade.
    up = lambda value, default: round(max(value * margin, default), 4)
    suggested = RuleConfig(
        **{**asdict(cfg),
           "shake_amp": up(p("yaw_amp", percentile), cfg.shake_amp),
           "nod_amp": up(p("pitch_amp", percentile), cfg.nod_amp),
           "brow_up": up(p("brow_altura", percentile), cfg.brow_up),
           "brow_down": up(-p("brow_altura", 100 - percentile), cfg.brow_down),
           "furrow": up(-p("brow_juntas", 100 - percentile), cfg.furrow)}
    )
    rate = {name: float(fired[i] / total) for i, name in enumerate(CLASS_NAMES)}
    return suggested, rate, total


def _cli():
    ap = argparse.ArgumentParser(description="Detector de marcadores faciais por regras.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="mede em .parquet SEM expressão e sugere limiares")
    c.add_argument("--features-dir", required=True)
    c.add_argument("--percentile", type=float, default=99.0)
    c.add_argument("--rules", default=None, help="json de limiares atual (opcional)")
    c.add_argument("--write", default=None, help="salva os limiares sugeridos neste json")

    v = sub.add_parser("video", help="roda as regras em um vídeo (ex.: você fazendo cada expressão)")
    v.add_argument("path")
    v.add_argument("--rules", default=None)
    v.add_argument("--neutral-seconds", type=float, default=2.0,
                   help="segundos iniciais do vídeo com rosto e cabeça parados (linha de base)")

    args = ap.parse_args()
    cfg = RuleConfig.load(args.rules) if args.rules else RuleConfig()

    if args.cmd == "calibrate":
        from pipeline.model.dataset import load_all_samples

        samples = load_all_samples(args.features_dir)
        suggested, rate, total = calibrate([s.sequence for s in samples], cfg, args.percentile)
        print(f"{len(samples)} clipes, {total} janelas")
        print("\nDisparos por engano com os limiares ATUAIS (fração das janelas):")
        for name, r in rate.items():
            if name != "NEUTRO":
                print(f"  {name}: {r:.1%}")
        print("\nLimiares sugeridos:")
        for k, val in asdict(suggested).items():
            if k in ("shake_amp", "nod_amp", "brow_up", "brow_down", "furrow"):
                print(f"  {k}: {val}")
        if args.write:
            suggested.save(args.write)
            print(f"\nsalvo em {args.write}")
    else:
        from pipeline.feature_extraction.landmark_extractor import extract_landmarks
        from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames

        pre = PreprocessConfig()
        frames = extract_frames(args.path, pre)
        seq = extract_landmarks(frames)
        fps = pre.target_fps or 25.0
        ts = [int(round(i / fps * 1000)) for i in range(len(frames))]
        cfg.fps = fps
        # Linha de base = os primeiros segundos parados. Sem isso, uma
        # expressão mantida o vídeo todo viraria "o normal" e sumiria.
        n0 = max(int(args.neutral_seconds * fps), cfg.min_frames)
        baseline = compute_baseline([base_features(seq[:n0])])
        segs = RuleExpressionPredictor(cfg).predict_segments(seq, ts, baseline=baseline)
        print(f"{len(frames)} quadros ({len(frames) / fps:.1f} s)")
        if not segs:
            print("nenhum marcador detectado")
        for s in segs:
            print(f"  {s.marker}: {s.start_ms / 1000:.1f}s -> {s.end_ms / 1000:.1f}s (confiança {s.confidence:.2f})")


if __name__ == "__main__":
    _cli()