"""
train_expressions.py
=====================
Treina o classificador de marcadores faciais (INTERROGATIVA_QU,
INTERROGATIVA_SN, NEGACAO, AFIRMACAO, NEUTRO) e mede com
leave-one-signer-out, como o classificador de sinais.

Entrada: a MESMA pasta de .parquet que `build_dataset_features` gera, só que
a partir de vídeos de expressão. Convenção das gravações:

    annotations.csv:  video_id, class, user_id
                      class   = o marcador (INTERROGATIVA_QU, NEGACAO, ...)
                      user_id = quem gravou
    Cada pessoa precisa gravar também clipes NEUTRO: eles dão a linha de
    base dela (sobrancelha, boca e postura em repouso).

Uso:
    python -m pipeline.model.train_expressions --features-dir dataset/features_expr --epochs 40 --output expression_model.pt
    # só medir, sem treinar o modelo final:
    python -m pipeline.model.train_expressions --features-dir ... --skip-final
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.feature_extraction.expression_features import (
    base_features,
    compute_baseline,
    fill_missing,
    sequence_features,
)
from pipeline.model.dataset import SignSample, leave_one_signer_out_splits, load_all_samples
from pipeline.model.expression_classifier import ExpressionClassifier, ExpressionConfig, save_checkpoint
from pipeline.model.expression_windows import sliding_windows

NEUTRAL = "NEUTRO"


def signer_baselines(samples: list[SignSample]) -> dict[str, np.ndarray]:
    """Linha de base por pessoa = mediana dos frames dos clipes NEUTRO dela.
    Quem não gravou NEUTRO fica sem entrada (usa a mediana do próprio clipe)."""
    per_signer: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in samples:
        if s.label == NEUTRAL:
            per_signer[s.signer_id].append(fill_missing(base_features(s.sequence)))
    return {signer: compute_baseline(seqs) for signer, seqs in per_signer.items()}


def build_windows(samples, class_to_idx, baselines, config: ExpressionConfig):
    xs, ys = [], []
    for s in samples:
        feats = sequence_features(s.sequence, baseline=baselines.get(s.signer_id), fps=config.fps)
        if len(feats) < config.win:  # clipe curto: repete o último frame
            feats = np.pad(feats, ((0, config.win - len(feats)), (0, 0)), mode="edge")
        windows, _ = sliding_windows(feats, config.win, config.stride)
        xs.append(windows)
        ys.extend([class_to_idx[s.label]] * len(windows))
    return torch.from_numpy(np.concatenate(xs)).float(), torch.tensor(ys, dtype=torch.long)


def train_model(x, y, config: ExpressionConfig, epochs: int, batch_size: int, lr: float, device: str):
    model = ExpressionClassifier(config).to(device)
    counts = Counter(y.tolist())
    weights = torch.tensor(
        [len(y) / (config.num_classes * max(counts.get(c, 0), 1)) for c in range(config.num_classes)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict(model, x, device: str) -> np.ndarray:
    return model(x.to(device)).argmax(dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Treino e validação (leave-one-signer-out) dos marcadores faciais.")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fps", type=float, default=25.0, help="25 / frame_stride usado ao gerar os parquet")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default="expression_model.pt")
    ap.add_argument("--skip-final", action="store_true", help="só valida; não treina o modelo final")
    args = ap.parse_args()

    samples = load_all_samples(args.features_dir)
    class_names = sorted({s.label for s in samples})
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    config = ExpressionConfig(num_classes=len(class_names), win=args.win, stride=args.stride, fps=args.fps)
    print(f"{len(samples)} clipes, classes: {class_names}")
    for c in class_names:
        print(f"  {c}: {sum(s.label == c for s in samples)} clipes")

    n_signers = len({s.signer_id for s in samples})
    if n_signers < 2:
        raise SystemExit("Preciso de pelo menos 2 pessoas pra validar leave-one-signer-out.")

    # Linha de base de cada pessoa vem só dos clipes dela (inclusive no teste:
    # na inferência ao vivo ela vem da própria pessoa, via RollingBaseline).
    baselines = signer_baselines(samples)
    missing = {s.signer_id for s in samples} - set(baselines)
    if missing:
        print(f"AVISO: sem clipes NEUTRO para {sorted(missing)}; usando a mediana do próprio clipe.")

    all_true, all_pred = [], []
    for signer, train_idx, test_idx in leave_one_signer_out_splits(samples):
        x_tr, y_tr = build_windows([samples[i] for i in train_idx], class_to_idx, baselines, config)
        x_te, y_te = build_windows([samples[i] for i in test_idx], class_to_idx, baselines, config)
        model = train_model(x_tr, y_tr, config, args.epochs, args.batch_size, args.lr, args.device)
        pred = predict(model, x_te, args.device)
        acc = float((pred == y_te.numpy()).mean())
        print(f"[{signer}] acurácia por janela: {acc:.1%} ({len(y_te)} janelas)")
        all_true.extend(y_te.tolist())
        all_pred.extend(pred.tolist())

    all_true, all_pred = np.array(all_true), np.array(all_pred)
    print(f"\nAgregada: {(all_true == all_pred).mean():.1%} (acaso: {1 / len(class_names):.1%})")
    for c, name in enumerate(class_names):
        mask = all_true == c
        if mask.any():
            wrong = Counter(all_pred[mask & (all_pred != c)].tolist()).most_common(1)
            conf = f", mais confundida com {class_names[wrong[0][0]]}" if wrong else ""
            print(f"  {name}: {(all_pred[mask] == c).mean():.0%}{conf}")

    if args.skip_final:
        return
    x, y = build_windows(samples, class_to_idx, baselines, config)
    model = train_model(x, y, config, args.epochs, args.batch_size, args.lr, args.device)
    save_checkpoint(model, class_names, args.output, extra={"epochs": args.epochs, "n_clips": len(samples)})
    print(f"\nModelo final (100% dos dados) salvo em {args.output}")


if __name__ == "__main__":
    main()