"""
prepare_ufop_dataset.py
=========================
Ingestão dos datasets da UFOP baixados manualmente do Zenodo (acesso
restrito, exige Dataset Usage Agreement — não dá pra automatizar esse
passo). Lê os .zip originais (subject_1..5.zip = sinais em pares
mínimos; LIBRAS-FACIAL-EXPRESIONS-BSL_ORIGINAL.zip = expressões
faciais) e gera, para cada um, uma pasta no formato que
`build_dataset_features.py` já espera (`annotations.csv` + `videos/`).

Só extrai o vídeo de cor de cada amostra (`color.mp4` / `Color.avi`) —
ignora os vídeos de profundidade e os `.txt` de esqueleto do Kinect, do
mesmo jeito que `download_minds_libras.py` já ignora landmarks
pré-processados: a pipeline sempre extrai os próprios landmarks via
MediaPipe, pra manter o mesmo espaço de features em todo o dataset.
Isso evita extrair ~90% do conteúdo de cada .zip (profundidade é a
maior parte do volume) — não precisa descompactar tudo pra depois
apagar, só pegamos o que vamos usar.

Os rótulos (nomes de pasta em inglês) são traduzidos pra português via
`ufop_label_translation.py`, pra bater com o resto do léxico (`Aluno`,
`Banco`, ...) e com o contrato de I4 (`CONTRATO_I4.md`, campo `sinal`).

Uso:
    python -m pipeline.ingestion.prepare_ufop_dataset \
        --subject-zips subject_1.zip subject_2.zip subject_3.zip subject_4.zip subject_5.zip \
        --output-dir dataset/ufop-sinais

    python -m pipeline.ingestion.prepare_ufop_dataset \
        --facial-zip LIBRAS-FACIAL-EXPRESIONS-BSL_ORIGINAL.zip \
        --output-dir dataset/ufop-expressoes-faciais
"""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from pipeline.ingestion.ufop_label_translation import (
    translate_facial_expression,
    translate_sign,
)

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_]+")


def _safe(part: str) -> str:
    return _SAFE_CHARS.sub("", part.replace(" ", "_"))


def ingest_signs(subject_zips: list[Path], output_dir: Path, overwrite: bool = False) -> int:
    """subject_N.zip: subject_N/<SinalIngles>/sample_XXX/color.mp4 (+ depth.mp4, skel/*.txt — ignorados)."""
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for zip_path in subject_zips:
        subject = zip_path.stem  # "subject_1"
        with zipfile.ZipFile(zip_path) as zf:
            color_entries = [n for n in zf.namelist() if n.lower().endswith("/color.mp4")]
            for i, entry in enumerate(color_entries, 1):
                parts = entry.split("/")
                # subject_1/Always/sample_001/color.mp4 -> parts[1]=Always, parts[2]=sample_001
                english_label, sample = parts[1], parts[2]
                sinal_pt = translate_sign(english_label)
                video_id = f"{subject}_{_safe(sinal_pt)}_{_safe(sample)}"
                dest = videos_dir / f"{video_id}.mp4"
                if not dest.exists() or overwrite:
                    with zf.open(entry) as src, open(dest, "wb") as out:
                        out.write(src.read())
                rows.append({"video_id": video_id, "class": sinal_pt, "user_id": subject})
                if i % 100 == 0 or i == len(color_entries):
                    print(f"  [{subject}] {i}/{len(color_entries)} vídeos extraídos")

    pd.DataFrame(rows).to_csv(output_dir / "annotations.csv", index=False)
    print(f"\n{len(rows)} amostras de sinais prontas em {output_dir.resolve()}")
    return len(rows)


def ingest_facial_expressions(facial_zip: Path, output_dir: Path, overwrite: bool = False) -> int:
    """LIBRAS-FACIAL-EXPRESIONS-BSL_ORIGINAL.zip:
    .../SUBJECT_NN/<idx_Expressao>/<sample_dir>/Color.avi (+ Depth/*.png — ignorados).
    Remuxa cada Color.avi pra .mp4 (sem recodificar) via ffmpeg, pra reusar
    build_dataset_features.py sem modificação (ele assume extensão .mp4)."""
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with zipfile.ZipFile(facial_zip) as zf:
        color_entries = [n for n in zf.namelist() if n.lower().endswith("/color.avi")]
        for i, entry in enumerate(color_entries, 1):
            parts = entry.split("/")
            # .../SUBJECT_01/0_Angry/0_p1_c4_s1_1/Color.avi
            subject, english_expr, sample = parts[-4], parts[-3], parts[-2]
            expr_pt = translate_facial_expression(english_expr)
            video_id = f"{subject}_{_safe(expr_pt)}_{_safe(sample)}"
            dest_avi = videos_dir / f"{video_id}.avi"
            dest_mp4 = videos_dir / f"{video_id}.mp4"
            if not dest_mp4.exists() or overwrite:
                with zf.open(entry) as src, open(dest_avi, "wb") as out:
                    out.write(src.read())
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(dest_avi), "-c", "copy", str(dest_mp4)],
                    check=True, capture_output=True,
                )
                dest_avi.unlink()
            rows.append({"video_id": video_id, "class": expr_pt, "user_id": subject})
            if i % 100 == 0 or i == len(color_entries):
                print(f"  {i}/{len(color_entries)} vídeos remuxados")

    pd.DataFrame(rows).to_csv(output_dir / "annotations.csv", index=False)
    print(f"\n{len(rows)} amostras de expressões faciais prontas em {output_dir.resolve()}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Ingestão dos datasets da UFOP (Zenodo).")
    parser.add_argument("--subject-zips", nargs="+", type=Path, default=None)
    parser.add_argument("--facial-zip", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.subject_zips:
        ingest_signs(args.subject_zips, args.output_dir, overwrite=args.overwrite)
    elif args.facial_zip:
        ingest_facial_expressions(args.facial_zip, args.output_dir, overwrite=args.overwrite)
    else:
        parser.error("passe --subject-zips ou --facial-zip")


if __name__ == "__main__":
    main()
