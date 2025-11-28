"""
Evaluation metrics module for ranking tasks.

This module provides utilities to compute ranking evaluation metrics:
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (nDCG)
- Mean Reciprocal Rank (MRR)
- Recall@K
- Precision@K
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


class RankingMetrics:
    """
    A class to compute ranking evaluation metrics.
    
    Supports:
    - MAP (Mean Average Precision)
    - nDCG (Normalized Discounted Cumulative Gain)
    - MRR (Mean Reciprocal Rank)
    - Recall@K
    - Precision@K
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    @staticmethod
    def precision_at_k(
        relevance: List[int], 
        k: int
    ) -> float:
        """
        Compute Precision@K for a single query.
        
        Args:
            relevance: List of relevance scores (0 or 1) sorted by predicted rank
            k: Number of top results to consider
            
        Returns:
            Precision@K score
        """
        if k <= 0:
            return 0.0
        
        relevance = relevance[:k]
        if len(relevance) == 0:
            return 0.0
        
        return sum(relevance) / min(k, len(relevance))
    
    @staticmethod
    def recall_at_k(
        relevance: List[int], 
        k: int, 
        total_relevant: Optional[int] = None
    ) -> float:
        """
        Compute Recall@K for a single query.
        
        Args:
            relevance: List of relevance scores (0 or 1) sorted by predicted rank
            k: Number of top results to consider
            total_relevant: Total number of relevant documents (if None, uses sum of relevance)
            
        Returns:
            Recall@K score
        """
        if k <= 0:
            return 0.0
        
        if total_relevant is None:
            total_relevant = sum(relevance)
        
        if total_relevant == 0:
            return 0.0
        
        relevant_at_k = sum(relevance[:k])
        return relevant_at_k / total_relevant
    
    @staticmethod
    def average_precision(relevance: List[int]) -> float:
        """
        Compute Average Precision for a single query.
        
        Args:
            relevance: List of relevance scores (0 or 1) sorted by predicted rank
            
        Returns:
            Average Precision score
        """
        if not relevance or sum(relevance) == 0:
            return 0.0
        
        precisions = []
        num_relevant = 0
        
        for i, rel in enumerate(relevance):
            if rel == 1:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precisions.append(precision_at_i)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / sum(relevance)
    
    @staticmethod
    def dcg_at_k(relevance: List[int], k: int) -> float:
        """
        Compute Discounted Cumulative Gain at K.
        
        Args:
            relevance: List of relevance scores sorted by predicted rank
            k: Number of top results to consider
            
        Returns:
            DCG@K score
        """
        relevance = relevance[:k]
        if len(relevance) == 0:
            return 0.0
        
        dcg = 0.0
        for i, rel in enumerate(relevance):
            # Using log2(i+2) to avoid log(1)=0
            dcg += rel / np.log2(i + 2)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(relevance: List[int], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K.
        
        Args:
            relevance: List of relevance scores sorted by predicted rank
            k: Number of top results to consider
            
        Returns:
            nDCG@K score
        """
        dcg = RankingMetrics.dcg_at_k(relevance, k)
        
        # Compute ideal DCG (perfect ranking)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = RankingMetrics.dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def reciprocal_rank(relevance: List[int]) -> float:
        """
        Compute Reciprocal Rank for a single query.
        
        Args:
            relevance: List of relevance scores (0 or 1) sorted by predicted rank
            
        Returns:
            Reciprocal Rank score
        """
        for i, rel in enumerate(relevance):
            if rel == 1:
                return 1.0 / (i + 1)
        return 0.0
    
    def compute_metrics(
        self,
        grouped_predictions: Dict[int, Tuple[List[float], List[int]]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute all ranking metrics for grouped predictions.
        
        Args:
            grouped_predictions: Dict mapping query_id to (predicted_scores, true_relevance)
            k_values: List of K values for @K metrics
            
        Returns:
            Dictionary of metric names to values
        """
        all_aps = []
        all_rrs = []
        recall_at_k = {k: [] for k in k_values}
        precision_at_k = {k: [] for k in k_values}
        ndcg_at_k = {k: [] for k in k_values}
        
        for query_id, (scores, relevance) in grouped_predictions.items():
            # Sort by predicted scores (descending)
            sorted_indices = np.argsort(scores)[::-1]
            sorted_relevance = [relevance[i] for i in sorted_indices]
            
            # Average Precision
            ap = self.average_precision(sorted_relevance)
            all_aps.append(ap)
            
            # Reciprocal Rank
            rr = self.reciprocal_rank(sorted_relevance)
            all_rrs.append(rr)
            
            # @K metrics
            for k in k_values:
                recall_at_k[k].append(
                    self.recall_at_k(sorted_relevance, k)
                )
                precision_at_k[k].append(
                    self.precision_at_k(sorted_relevance, k)
                )
                ndcg_at_k[k].append(
                    self.ndcg_at_k(sorted_relevance, k)
                )
        
        # Aggregate metrics
        metrics = {
            'MAP': np.mean(all_aps) if all_aps else 0.0,
            'MRR': np.mean(all_rrs) if all_rrs else 0.0
        }
        
        for k in k_values:
            metrics[f'Recall@{k}'] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0.0
            metrics[f'Precision@{k}'] = np.mean(precision_at_k[k]) if precision_at_k[k] else 0.0
            metrics[f'nDCG@{k}'] = np.mean(ndcg_at_k[k]) if ndcg_at_k[k] else 0.0
        
        return metrics
    
    def evaluate_model(
        self,
        model,
        feature_extractor,
        test_data: pd.DataFrame,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate a ranking model on test data.
        
        Args:
            model: Trained RankingModel
            feature_extractor: Fitted FeatureExtractor
            test_data: Test DataFrame with query, tafsir_text, relevance columns
            k_values: List of K values for @K metrics
            
        Returns:
            Dictionary of metric names to values
        """
        if 'query_id' not in test_data.columns:
            test_data = test_data.copy()
            test_data['query_id'] = pd.factorize(test_data['query'])[0]
        
        grouped_predictions = {}
        
        for query_id, group in test_data.groupby('query_id'):
            queries = group['query'].tolist()
            documents = group['tafsir_text'].tolist()
            relevance = group['relevance'].tolist()
            
            # Extract features and predict scores
            features = feature_extractor.extract_features(queries, documents)
            scores = model.predict_scores(features).tolist()
            
            grouped_predictions[query_id] = (scores, relevance)
        
        return self.compute_metrics(grouped_predictions, k_values)


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Pretty print ranking metrics.
    
    Args:
        metrics: Dictionary of metric names to values
    """
    print("\n" + "=" * 50)
    print("Ranking Evaluation Metrics")
    print("=" * 50)
    
    # Print main metrics
    print(f"\nMAP (Mean Average Precision): {metrics.get('MAP', 0):.4f}")
    print(f"MRR (Mean Reciprocal Rank):   {metrics.get('MRR', 0):.4f}")
    
    # Print @K metrics
    k_values = sorted(set(
        int(k.split('@')[1]) 
        for k in metrics.keys() 
        if '@' in k
    ))
    
    if k_values:
        print("\n@K Metrics:")
        print("-" * 40)
        print(f"{'K':<5} {'Recall':<12} {'Precision':<12} {'nDCG':<12}")
        print("-" * 40)
        
        for k in k_values:
            recall = metrics.get(f'Recall@{k}', 0)
            precision = metrics.get(f'Precision@{k}', 0)
            ndcg = metrics.get(f'nDCG@{k}', 0)
            print(f"{k:<5} {recall:<12.4f} {precision:<12.4f} {ndcg:<12.4f}")
    
    print("=" * 50 + "\n")
