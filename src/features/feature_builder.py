# src/features/feature_builder.py
from __future__ import annotations

from typing import List
from pathlib import Path
import re
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# =========================================
# GLOBAL: path & artefak feature
# =========================================
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"

_ARTEFACTS: dict | None = None
_VECTORIZER: TfidfVectorizer | None = None
_E5_MODEL: SentenceTransformer | None = None
FEATURE_COLS: List[str] = []   # akan diisi dari artefak


def _load_artefacts() -> dict:
    """
    Load feature_artefacts.pkl satu kali, cache di global.
    Isi tipikal: {
        'tfidf_vectorizer': <TfidfVectorizer>,
        'sbert_model_name': 'intfloat/multilingual-e5-base',
        'feature_names': ['feat_tfidf_similarity', 'feat_sbert_similarity', 'feat_keyword_overlap']
    }
    """
    global _ARTEFACTS, FEATURE_COLS
    if _ARTEFACTS is None:
        path = MODELS_DIR / "feature_artefacts.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"feature_artefacts.pkl tidak ditemukan di {path}. "
                "Pastikan file ini sudah disalin dari Kaggle ke folder models/."
            )
        with open(path, "rb") as f:
            _ARTEFACTS = pickle.load(f)

        # set FEATURE_COLS dari artefak agar konsisten dengan training
        feat_names = _ARTEFACTS.get("feature_names", [])
        if not feat_names:
            raise ValueError("feature_artefacts.pkl tidak memiliki key 'feature_names'.")
        FEATURE_COLS.clear()
        FEATURE_COLS.extend(feat_names)

    return _ARTEFACTS


def _get_vectorizer() -> TfidfVectorizer:
    """
    Ambil TF-IDF vectorizer yang sudah di-fit dari artefak.
    Tidak di-fit ulang.
    """
    global _VECTORIZER
    if _VECTORIZER is None:
        arte = _load_artefacts()
        vec = arte.get("tfidf_vectorizer", None)
        if vec is None:
            raise ValueError("Artefak tidak mengandung 'tfidf_vectorizer'.")
        _VECTORIZER = vec
    return _VECTORIZER


def _get_e5_model() -> SentenceTransformer:
    """
    Load SentenceTransformer berdasarkan 'sbert_model_name' di artefak.
    """
    global _E5_MODEL
    if _E5_MODEL is None:
        arte = _load_artefacts()
        model_name = arte.get("sbert_model_name", "intfloat/multilingual-e5-base")
        _E5_MODEL = SentenceTransformer(model_name)
    return _E5_MODEL

# =========================================
# Helper: preprocessing dan similarity
# =========================================

def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\u0600-\u06FF]+", " ", text)
    tokens = text.split()
    return tokens


def _keyword_overlap(query: str, docs: List[str]) -> np.ndarray:
    q_tokens = set(_simple_tokenize(query))
    if not q_tokens:
        return np.zeros(len(docs), dtype=float)

    overlaps = []
    for t in docs:
        d_tokens = set(_simple_tokenize(t))
        if not d_tokens:
            overlaps.append(0.0)
            continue
        inter = len(q_tokens & d_tokens)
        union = len(q_tokens | d_tokens)
        overlaps.append(inter / union)
    return np.asarray(overlaps, dtype=float)


def _tfidf_similarity(
    query: str, docs: List[str], vectorizer: TfidfVectorizer
) -> np.ndarray:
    q_vec = vectorizer.transform([query])
    d_vec = vectorizer.transform(docs)

    q_data = q_vec.toarray()[0]
    d_data = d_vec.toarray()

    q_norm = np.linalg.norm(q_data) + 1e-9
    d_norm = np.linalg.norm(d_data, axis=1) + 1e-9
    sims = (d_data @ q_data) / (d_norm * q_norm)
    return sims


def _sbert_similarity(
    query: str, docs: List[str], model: SentenceTransformer
) -> np.ndarray:
    emb_q = model.encode([query], normalize_embeddings=True)
    emb_d = model.encode(docs, normalize_embeddings=True)
    sims = (emb_d @ emb_q[0].T)
    return sims.ravel()

# =========================================
# FUNGSI UTAMA: build fitur
# =========================================

def build_features_for_pairs(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    df_pairs harus punya kolom:
      - 'query'
      - 'tafsir'
    Output: DataFrame dengan kolom sesuai FEATURE_COLS dari artefak.
    """
    if "query" not in df_pairs.columns or "tafsir" not in df_pairs.columns:
        raise ValueError("df_pairs harus punya kolom 'query' dan 'tafsir'.")

    _load_artefacts()  # pastikan FEATURE_COLS terisi

    queries = df_pairs["query"].astype(str).tolist()
    tafsirs = df_pairs["tafsir"].astype(str).tolist()

    query_text = queries[0]  # asumsi 1 query vs banyak tafsir

    vectorizer = _get_vectorizer()
    e5_model = _get_e5_model()

    tfidf_sims = _tfidf_similarity(query_text, tafsirs, vectorizer)
    sbert_sims = _sbert_similarity(query_text, tafsirs, e5_model)
    keyword_sims = _keyword_overlap(query_text, tafsirs)

    # mapping fitur berdasarkan nama (supaya fleksibel)
    data = {}
    for name in FEATURE_COLS:
        lname = name.lower()
        if "tfidf" in lname:
            data[name] = tfidf_sims
        elif "sbert" in lname or "e5" in lname:
            data[name] = sbert_sims
        elif "keyword" in lname or "overlap" in lname or "jaccard" in lname:
            data[name] = keyword_sims
        else:
            # fallback: pakai tfidf_sims (biar nggak crash kalau ada nama aneh)
            data[name] = tfidf_sims

    df_feat = pd.DataFrame(data, index=df_pairs.index)
    return df_feat


def build_features_for_query_corpus(
    query_text: str, df_corpus: pd.DataFrame
) -> pd.DataFrame:
    """
    Bentuk semua pasangan (query, tafsir_i) untuk seluruh baris df_corpus,
    lalu panggil build_features_for_pairs.
    """
    if "tafsir" not in df_corpus.columns:
        raise ValueError("df_corpus harus punya kolom 'tafsir'.")

    df_pairs = df_corpus.copy().reset_index(drop=True)
    df_pairs["query"] = query_text

    df_feat = build_features_for_pairs(df_pairs)
    return df_feat
