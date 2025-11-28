"""
Quran Query-Tafsir Ranking ML Package

This package provides utilities for:
- Loading and preprocessing tafsir data
- Feature extraction (TF-IDF, SBERT embeddings)
- Model training (Logistic Regression, SVM, XGBoost)
- Evaluation metrics (MAP, nDCG, MRR, Recall@K)
"""

from .data_loader import TafsirDataLoader
from .features import FeatureExtractor
from .models import RankingModel
from .evaluation import RankingMetrics

__version__ = "0.1.0"
__all__ = ["TafsirDataLoader", "FeatureExtractor", "RankingModel", "RankingMetrics"]
