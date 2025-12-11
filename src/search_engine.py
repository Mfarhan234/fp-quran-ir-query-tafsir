import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import string
import nltk
from nltk.corpus import stopwords
import torch
import os
import sys
import pickle

# Setup NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class QuranSearchEngine:
    def __init__(self, base_dir=None):
        """
        Engine pencarian utama.
        base_dir: Path root project. Jika None, otomatis deteksi.
        """
        if base_dir is None:
            # Otomatis naik 2 level dari src/ ke root
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
        
        # 1. Definisi Path (Sesuai struktur folder Anda)
        self.models_dir = os.path.join(self.base_dir, 'models')
        
        # Folder & File Model
        self.sbert_path = os.path.join(self.models_dir, 'sbert_finetuned_quran')
        self.xgb_path = os.path.join(self.models_dir, 'xgboost_best_model.json')
        self.threshold_path = os.path.join(self.models_dir, 'threshold.txt')
        
        # File Cache (Yang barusan Anda simpan)
        self.cache_emb_path = os.path.join(self.models_dir, 'corpus_embeddings.pt')
        self.cache_data_path = os.path.join(self.models_dir, 'corpus_data.pkl')

        # Variables
        self.sbert = None
        self.xgb_model = None
        self.threshold = 0.3 # Default
        self.corpus_texts = []
        self.metadata_map = {}
        self.corpus_embeddings = None
        self.bm25 = None
        self.stop_words = set(stopwords.words('indonesian'))
        
        # Fitur yang dipakai XGBoost (Urutan PENTING)
        self.features = ['sbert_sim', 'bm25_score', 'overlap_score', 'jaccard_score']

        # Mulai Loading
        self._load_resources()

    def _clean_tokens(self, text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        return [w for w in tokens if w not in self.stop_words]

    def _load_resources(self):
        print("[INFO] Memulai inisialisasi Search Engine...")

        # A. Load Model SBERT & XGBoost
        if not os.path.exists(self.sbert_path):
            raise FileNotFoundError(f"Model SBERT tidak ditemukan di {self.sbert_path}")
        
        try:
            self.sbert = SentenceTransformer(self.sbert_path, trust_remote_code=True)
        except:
            self.sbert = SentenceTransformer(self.sbert_path)
            
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(self.xgb_path)
        
        if os.path.exists(self.threshold_path):
            with open(self.threshold_path, 'r') as f:
                self.threshold = float(f.read().strip())
        
        # B. Load Cache Data (Embeddings & Teks)
        if os.path.exists(self.cache_emb_path) and os.path.exists(self.cache_data_path):
            print("   -> [FAST LOAD] Memuat Embeddings & Data dari Cache...")
            
            # 1. Load Tensor
            self.corpus_embeddings = torch.load(self.cache_emb_path)
            
            # 2. Load Data Dictionary
            with open(self.cache_data_path, 'rb') as f:
                cache_data = pickle.load(f)
                # Ambil data dari dictionary yang kita simpan tadi
                self.corpus_texts = cache_data.get('texts', [])
                # Fallback nama key (jika tadi nyimpannya beda nama)
                if not self.corpus_texts: self.corpus_texts = cache_data.get('unique_tafsirs', [])
                
                self.metadata_map = cache_data.get('metadata', {})
                if not self.metadata_map: self.metadata_map = cache_data.get('metadata_map', {})
            
            print(f"   -> Berhasil memuat {len(self.corpus_texts)} dokumen tafsir.")
        else:
            raise FileNotFoundError(
                "❌ File Cache tidak ditemukan! Harap jalankan proses indexing/saving di Notebook dulu."
            )

        # C. Build BM25 Index (Cepat, build on-the-fly)
        print("   -> Membangun Index BM25...")
        corpus_tokens = [self._clean_tokens(doc) for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(corpus_tokens)
        
        print("[INFO] Engine Siap Digunakan.")

    def search(self, query, top_k=10):
        """
        Melakukan pencarian hybrid.
        """
        # 1. Retrieval (Semantic)
        query_vec = self.sbert.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_vec, self.corpus_embeddings, top_k=50)[0]
        
        # 2. Feature Extraction
        candidates = []
        q_toks = self._clean_tokens(query)
        
        for hit in hits:
            idx = hit['corpus_id']
            txt = self.corpus_texts[idx]
            
            # Hitung Fitur Lexical
            t_toks = self._clean_tokens(txt)
            
            set_q, set_t = set(q_toks), set(t_toks)
            overlap = len(set_q.intersection(set_t)) / len(set_q) if len(set_q) > 0 else 0
            jaccard = len(set_q.intersection(set_t)) / (len(set_q.union(set_t)) + 1e-9)
            bm25_score = self.bm25.get_batch_scores(q_toks, [idx])[0]
            
            candidates.append({
                'text': txt,
                'sbert_sim': hit['score'],
                'bm25_score': bm25_score,
                'overlap_score': overlap,
                'jaccard_score': jaccard
            })
            
        if not candidates:
            return []

        # 3. Re-Ranking (XGBoost)
        df_cand = pd.DataFrame(candidates)
        dtest = xgb.DMatrix(df_cand[self.features])
        probs = self.xgb_model.predict(dtest)
        df_cand['final_score'] = probs
        
        # 4. Filter & Sort
        results = df_cand[df_cand['final_score'] > self.threshold].sort_values('final_score', ascending=False)
        
        # Format Output
        final_output = []
        for _, row in results.head(top_k).iterrows():
            txt_content = row['text']
            # Cari Lokasi di Metadata Map
            lokasi = self.metadata_map.get(txt_content.strip(), "Lokasi tidak diketahui")
            
            final_output.append({
                'lokasi': lokasi,
                'text': txt_content,
                'score': float(row['final_score']),
                'sbert': float(row['sbert_sim'])
            })
            
        return final_output