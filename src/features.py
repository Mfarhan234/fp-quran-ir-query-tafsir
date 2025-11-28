"""
Feature extraction module for query-tafsir ranking.

This module provides utilities to extract features from text using:
- TF-IDF vectorization
- SBERT (Sentence-BERT) embeddings
- Combined feature representations
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class FeatureExtractor:
    """
    A class to extract features from query-document pairs.
    
    Supports:
    - TF-IDF features
    - SBERT embeddings
    - Combined TF-IDF + SBERT features
    """
    
    def __init__(
        self,
        use_tfidf: bool = True,
        use_sbert: bool = True,
        sbert_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize the feature extractor.
        
        Args:
            use_tfidf: Whether to use TF-IDF features
            use_sbert: Whether to use SBERT embeddings
            sbert_model_name: Name of the SBERT model to use
        """
        self.use_tfidf = use_tfidf
        self.use_sbert = use_sbert
        self.sbert_model_name = sbert_model_name
        
        self.tfidf_vectorizer = None
        self.sbert_model = None
        self._sbert_loaded = False
        
        if use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
    
    def _load_sbert(self) -> None:
        """Load SBERT model lazily."""
        if not self._sbert_loaded and self.use_sbert:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer(self.sbert_model_name)
                self._sbert_loaded = True
            except ImportError:
                print("Warning: sentence-transformers not installed. SBERT features disabled.")
                self.use_sbert = False
    
    def fit_tfidf(self, texts: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on a corpus.
        
        Args:
            texts: List of text documents
        """
        if self.use_tfidf:
            self.tfidf_vectorizer.fit(texts)
    
    def get_tfidf_features(
        self, 
        queries: List[str], 
        documents: List[str]
    ) -> np.ndarray:
        """
        Extract TF-IDF-based features for query-document pairs.
        
        Features include:
        - Cosine similarity between query and document TF-IDF vectors
        - Query term coverage in document
        
        Args:
            queries: List of query strings
            documents: List of document strings
            
        Returns:
            NumPy array of TF-IDF features
        """
        if not self.use_tfidf:
            return np.array([]).reshape(len(queries), 0)
        
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")
        
        query_vectors = self.tfidf_vectorizer.transform(queries)
        doc_vectors = self.tfidf_vectorizer.transform(documents)
        
        features = []
        for i in range(len(queries)):
            q_vec = query_vectors[i]
            d_vec = doc_vectors[i]
            
            # Cosine similarity
            cos_sim = cosine_similarity(q_vec, d_vec)[0, 0]
            
            # Query term coverage
            q_terms = set(queries[i].lower().split())
            d_terms = set(documents[i].lower().split())
            coverage = len(q_terms & d_terms) / max(len(q_terms), 1)
            
            # Document length (normalized)
            doc_len = len(documents[i].split()) / 100.0
            
            features.append([cos_sim, coverage, doc_len])
        
        return np.array(features)
    
    def get_sbert_features(
        self, 
        queries: List[str], 
        documents: List[str]
    ) -> np.ndarray:
        """
        Extract SBERT embedding-based features for query-document pairs.
        
        Features include:
        - Cosine similarity between query and document embeddings
        - Concatenated query and document embeddings (optional)
        
        Args:
            queries: List of query strings
            documents: List of document strings
            
        Returns:
            NumPy array of SBERT features
        """
        if not self.use_sbert:
            return np.array([]).reshape(len(queries), 0)
        
        self._load_sbert()
        
        if self.sbert_model is None:
            return np.array([]).reshape(len(queries), 0)
        
        # Encode queries and documents
        query_embeddings = self.sbert_model.encode(queries, show_progress_bar=False)
        doc_embeddings = self.sbert_model.encode(documents, show_progress_bar=False)
        
        features = []
        for i in range(len(queries)):
            q_emb = query_embeddings[i].reshape(1, -1)
            d_emb = doc_embeddings[i].reshape(1, -1)
            
            # Cosine similarity
            cos_sim = cosine_similarity(q_emb, d_emb)[0, 0]
            
            # Element-wise product (interaction features)
            interaction = q_emb[0] * d_emb[0]
            
            # Combine features
            feat = np.concatenate([[cos_sim], interaction])
            features.append(feat)
        
        return np.array(features)
    
    def get_sbert_similarity_only(
        self, 
        queries: List[str], 
        documents: List[str]
    ) -> np.ndarray:
        """
        Get only SBERT cosine similarity scores (lightweight feature).
        
        Args:
            queries: List of query strings
            documents: List of document strings
            
        Returns:
            NumPy array of similarity scores
        """
        if not self.use_sbert:
            return np.zeros((len(queries), 1))
        
        self._load_sbert()
        
        if self.sbert_model is None:
            return np.zeros((len(queries), 1))
        
        query_embeddings = self.sbert_model.encode(queries, show_progress_bar=False)
        doc_embeddings = self.sbert_model.encode(documents, show_progress_bar=False)
        
        similarities = []
        for i in range(len(queries)):
            q_emb = query_embeddings[i].reshape(1, -1)
            d_emb = doc_embeddings[i].reshape(1, -1)
            cos_sim = cosine_similarity(q_emb, d_emb)[0, 0]
            similarities.append([cos_sim])
        
        return np.array(similarities)
    
    def extract_features(
        self, 
        queries: List[str], 
        documents: List[str],
        use_full_sbert: bool = False
    ) -> np.ndarray:
        """
        Extract combined features for query-document pairs.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            use_full_sbert: If True, use full SBERT features including interaction
            
        Returns:
            NumPy array of combined features
        """
        feature_list = []
        
        # TF-IDF features
        if self.use_tfidf:
            tfidf_feats = self.get_tfidf_features(queries, documents)
            if tfidf_feats.size > 0:
                feature_list.append(tfidf_feats)
        
        # SBERT features
        if self.use_sbert:
            if use_full_sbert:
                sbert_feats = self.get_sbert_features(queries, documents)
            else:
                sbert_feats = self.get_sbert_similarity_only(queries, documents)
            if sbert_feats.size > 0:
                feature_list.append(sbert_feats)
        
        if not feature_list:
            raise ValueError("No features extracted. Enable at least one feature type.")
        
        return np.hstack(feature_list)
    
    def save(self, filepath: str) -> None:
        """
        Save the feature extractor state.
        
        Args:
            filepath: Path to save the feature extractor
        """
        state = {
            'use_tfidf': self.use_tfidf,
            'use_sbert': self.use_sbert,
            'sbert_model_name': self.sbert_model_name,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        joblib.dump(state, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """
        Load a saved feature extractor.
        
        Args:
            filepath: Path to the saved feature extractor
            
        Returns:
            Loaded FeatureExtractor instance
        """
        state = joblib.load(filepath)
        extractor = cls(
            use_tfidf=state['use_tfidf'],
            use_sbert=state['use_sbert'],
            sbert_model_name=state['sbert_model_name']
        )
        extractor.tfidf_vectorizer = state['tfidf_vectorizer']
        return extractor
