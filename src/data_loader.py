"""
Data loader module for tafsir datasets.

This module provides utilities to load and preprocess tafsir data
for the query-tafsir ranking task.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union


class TafsirDataLoader:
    """
    A class to load and preprocess tafsir data for ranking tasks.
    
    The expected data format is a CSV or JSON file with columns:
    - query: The search query or question
    - tafsir_text: The tafsir explanation text
    - relevance: Relevance score (binary or graded)
    - surah (optional): Surah number
    - ayah (optional): Ayah number
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data directory or file
        """
        self.data_path = data_path
        self.data = None
        
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load tafsir data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing the tafsir data
        """
        self.data = pd.read_csv(filepath)
        self._validate_data()
        return self.data
    
    def load_json(self, filepath: str) -> pd.DataFrame:
        """
        Load tafsir data from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            DataFrame containing the tafsir data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        self.data = pd.DataFrame(json_data)
        self._validate_data()
        return self.data
    
    def _validate_data(self) -> None:
        """
        Validate that required columns exist in the data.
        """
        required_cols = ['query', 'tafsir_text', 'relevance']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def create_sample_data(self, n_queries: int = 10, n_docs_per_query: int = 5) -> pd.DataFrame:
        """
        Create sample tafsir data for demonstration purposes.
        
        Args:
            n_queries: Number of sample queries
            n_docs_per_query: Number of documents per query
            
        Returns:
            DataFrame with sample tafsir data
        """
        np.random.seed(42)
        
        sample_queries = [
            "Apa makna taqwa dalam Islam?",
            "Bagaimana hukum shalat fardhu?",
            "Apa hikmah puasa Ramadhan?",
            "Siapakah Nabi Ibrahim?",
            "Apa arti sabar menurut Al-Quran?",
            "Bagaimana cara bertaubat yang benar?",
            "Apa makna ihsan dalam Islam?",
            "Apa hukum zakat fitrah?",
            "Siapakah Maryam dalam Al-Quran?",
            "Apa makna syukur kepada Allah?"
        ]
        
        sample_tafsirs = [
            "Taqwa adalah memelihara diri dari siksa Allah dengan mengerjakan amal shalih dan meninggalkan maksiat.",
            "Shalat fardhu adalah ibadah wajib yang harus dikerjakan lima kali sehari oleh setiap muslim.",
            "Puasa Ramadhan merupakan rukun Islam yang ketiga dan diwajibkan bagi setiap muslim.",
            "Nabi Ibrahim adalah khalilullah (kekasih Allah) dan bapak para nabi.",
            "Sabar adalah menahan diri dari keluh kesah dan tetap tabah menghadapi cobaan.",
            "Taubat yang benar adalah menyesali perbuatan dosa, meninggalkannya, dan bertekad tidak mengulanginya.",
            "Ihsan adalah beribadah kepada Allah seolah-olah kamu melihat-Nya.",
            "Zakat fitrah wajib dikeluarkan sebelum shalat Idul Fitri sebagai penyuci jiwa.",
            "Maryam adalah wanita suci yang dipilih Allah untuk melahirkan Nabi Isa.",
            "Syukur adalah mengakui nikmat Allah dan menggunakannya sesuai perintah-Nya."
        ]
        
        data_rows = []
        for i, query in enumerate(sample_queries[:n_queries]):
            for j in range(n_docs_per_query):
                tafsir_idx = (i + j) % len(sample_tafsirs)
                relevance = 1 if j == 0 else np.random.randint(0, 2)
                data_rows.append({
                    'query_id': i,
                    'query': query,
                    'doc_id': i * n_docs_per_query + j,
                    'tafsir_text': sample_tafsirs[tafsir_idx],
                    'relevance': relevance,
                    'surah': np.random.randint(1, 115),
                    'ayah': np.random.randint(1, 20)
                })
        
        self.data = pd.DataFrame(data_rows)
        return self.data
    
    def get_query_document_pairs(self) -> Tuple[List[str], List[str], List[int]]:
        """
        Get query-document pairs and their relevance labels.
        
        Returns:
            Tuple of (queries, documents, relevance_labels)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv(), load_json(), or create_sample_data() first.")
        
        queries = self.data['query'].tolist()
        documents = self.data['tafsir_text'].tolist()
        relevance = self.data['relevance'].tolist()
        
        return queries, documents, relevance
    
    def get_grouped_by_query(self) -> Dict[int, pd.DataFrame]:
        """
        Group data by query_id for ranking evaluation.
        
        Returns:
            Dictionary mapping query_id to DataFrame of documents
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if 'query_id' not in self.data.columns:
            self.data['query_id'] = pd.factorize(self.data['query'])[0]
        
        return {qid: group for qid, group in self.data.groupby('query_id')}
    
    def train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets by query.
        
        Args:
            test_size: Proportion of queries for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if 'query_id' not in self.data.columns:
            self.data['query_id'] = pd.factorize(self.data['query'])[0]
        
        unique_queries = self.data['query_id'].unique()
        np.random.seed(random_state)
        np.random.shuffle(unique_queries)
        
        split_idx = int(len(unique_queries) * (1 - test_size))
        train_queries = unique_queries[:split_idx]
        test_queries = unique_queries[split_idx:]
        
        train_df = self.data[self.data['query_id'].isin(train_queries)]
        test_df = self.data[self.data['query_id'].isin(test_queries)]
        
        return train_df, test_df
