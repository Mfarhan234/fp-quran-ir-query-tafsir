# src/features/feature_builder.py
from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np

# Misal: daftar kolom fitur yang dipakai saat training
FEATURE_COLS = [
    "cosine_e5",
    "cosine_tfidf",
    "jaccard_overlap",
    # tambahkan fitur lain yang memang kalian pakai
]


def build_features_for_pairs(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    df_pairs: DataFrame yang minimal punya kolom:
      - 'query'   : teks query
      - 'tafsir'  : teks tafsir (atau kolom yang kalian pakai)
    Output: DataFrame fitur dengan kolom FEATURE_COLS.
    """
    # TODO: isi dengan logika feature engineering kalian
    # (TF-IDF, SBERT/E5, Jaccard, dsb.)
    # Contoh placeholder:
    feats = pd.DataFrame(index=df_pairs.index)

    # contoh dummy:
    feats["cosine_e5"] = 0.0
    feats["cosine_tfidf"] = 0.0
    feats["jaccard_overlap"] = 0.0

    return feats


def build_features_for_query_corpus(query_text: str, df_corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Bentuk semua pasangan (query, tafsir_i) untuk seluruh baris df_corpus,
    lalu panggil build_features_for_pairs.
    """
    df_pairs = df_corpus.copy()
    df_pairs = df_pairs.reset_index(drop=True)
    df_pairs["query"] = query_text

    df_feat = build_features_for_pairs(df_pairs)
    return df_feat
