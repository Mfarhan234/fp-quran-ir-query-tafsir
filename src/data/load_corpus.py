# src/data/load_corpus.py
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

def load_tafsir_corpus() -> pd.DataFrame:
    """
    Load korpus tafsir utama.
    Asumsi file: data/processed/tafsir_clean.csv
    Kolom minimal: surah, ayah, arabic_text, indonesian_translation, tafsir
    """
    path_processed = DATA_DIR / "processed" / "tafsir_clean.csv"
    if path_processed.exists():
        df = pd.read_csv(path_processed)
    else:
        # fallback kalau kamu pakai nama/struktur lain
        raise FileNotFoundError(
            f"tafsir_clean.csv tidak ditemukan di {path_processed}. "
            "Sesuaikan path di load_corpus.py."
        )

    # pastikan kolom penting ada (kalau beda nama, ubah di sini)
    expected_cols = ["surah", "ayah", "arabic_text", "indonesian_translation", "tafsir"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom berikut belum ada di tafsir_clean.csv: {missing}. "
            "Mapping-kan ke nama kolom yang benar di load_corpus.py."
        )
    return df
