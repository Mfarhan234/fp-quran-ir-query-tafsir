# src/models/relevance_model_loader.py
from pathlib import Path
import pickle
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from src.features.feature_builder import build_features_for_query_corpus


ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"

# Nama-nama model yang kita punya di folder models/
AVAILABLE_MODELS = [
    "XGBoost",
    "LightGBM",
    "LogisticRegression",
    "RandomForest",
    "SVM",
]


def _load_pickle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}. "
            "Pastikan sudah menyalin .pkl dari Kaggle ke folder models/."
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_best_model() -> Tuple[Any, Any, List[str], bool, str]:
    """
    Load best_relevance_model.pkl
    Return: model, scaler, feature_names, use_scaled, model_name
    """
    artefacts = _load_pickle(MODELS_DIR / "best_relevance_model.pkl")
    model = artefacts["model"]
    scaler = artefacts.get("scaler", None)
    feature_names = artefacts.get("feature_names", [])
    use_scaled = artefacts.get("use_scaled", True)
    name = artefacts.get("best_model_name", "UNKNOWN")
    return model, scaler, feature_names, use_scaled, name


def load_model_by_name(name: str) -> Tuple[Any, Any, List[str], bool]:
    """
    Load model_{name}.pkl, misal: 'XGBoost' -> models/model_XGBoost.pkl
    Return: model, scaler, feature_names, use_scaled
    """
    path = MODELS_DIR / f"model_{name}.pkl"
    artefacts = _load_pickle(path)
    model = artefacts["model"]
    scaler = artefacts.get("scaler", None)
    feature_names = artefacts.get("feature_names", [])
    use_scaled = artefacts.get("use_scaled", True)
    return model, scaler, feature_names, use_scaled


def score_query_against_corpus(
    query_text: str,
    df_corpus: pd.DataFrame,
    model,
    scaler=None,
    feature_names: list[str] | None = None,
    use_scaled: bool = True,
) -> np.ndarray:
    """
    Hitung skor relevansi 1 query terhadap seluruh df_corpus.
    Return: array skor (probabilitas) panjang = len(df_corpus)
    """
    df_feat = build_features_for_query_corpus(query_text, df_corpus)

    if feature_names:
        df_feat = df_feat[feature_names]

    X = df_feat.values

    if use_scaled and scaler is not None:
        X = scaler.transform(X)

    scores = model.predict_proba(X)[:, 1]
    return scores
