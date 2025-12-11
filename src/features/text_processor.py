import os
import string
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

class HybridEngine:
    def __init__(self, corpus_text, stopwords):
        """
        Inisialisasi Mesin Pencari Hybrid (SBERT + BM25).
        
        Args:
            corpus_text (list): List berisi teks tafsir (string).
            stopwords (set): Set kata sambung untuk preprocessing.
        """
        self.corpus = corpus_text
        self.stopwords = stopwords
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # ---------------------------------------------------------
        # 1. LOGIKA PEMILIHAN MODEL (LOKAL vs ONLINE)
        # ---------------------------------------------------------
        # Lokasi file ini: src/features/text_processor.py
        # Kita perlu naik 2 level ke root project untuk mencari folder 'models'
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        LOCAL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sbert_finetuned_quran')
        
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"Menggunakan SBERT Fine-Tuned LOKAL dari: {LOCAL_MODEL_PATH}")
            self.model_name = LOCAL_MODEL_PATH
        else:
            print("Folder Fine-Tuned tidak ditemukan di 'models/'.")
            print("Menggunakan Base Model (Online) sebagai fallback.")
            self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            
        # ---------------------------------------------------------
        # 2. LOAD SBERT & ENCODE CORPUS
        # ---------------------------------------------------------
        print("Sedang memuat SBERT & Encode Database Tafsir (Mohon tunggu)...")
        self.sbert = SentenceTransformer(self.model_name, device=self.device)
        
        # Mengubah ribuan teks tafsir menjadi vektor matematika
        # convert_to_tensor=True agar pencarian nanti bisa pakai GPU (cepat)
        self.corpus_embeddings = self.sbert.encode(self.corpus, convert_to_tensor=True, show_progress_bar=True)
        
        # ---------------------------------------------------------
        # 3. BANGUN INDEX BM25
        # ---------------------------------------------------------
        print("Sedang membangun Index BM25 (Kata Kunci)...")
        self.tokenized_corpus = [self.clean(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print("Hybrid Engine SIAP DIGUNAKAN!")

    def clean(self, text):
        """
        Membersihkan teks agar adil saat dihitung Overlap/BM25.
        - Lowercase
        - Hapus tanda baca
        - Hapus stopwords
        """
        text = str(text).lower()
        # Menghapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        # Filter stopwords
        return [t for t in tokens if t not in self.stopwords]

    def search_candidates(self, query, top_k=50):
        """
        Fungsi Utama Pencarian Kandidat.
        1. Mengambil top_k kandidat menggunakan SBERT (Retrieval).
        2. Menghitung 4 Fitur Inti (SBERT, BM25, Overlap, Jaccard) untuk XGBoost.
        """
        # --- TAHAP 1: RETRIEVAL (SBERT) ---
        query_vec = self.sbert.encode(query, convert_to_tensor=True)
        # Mencari dokumen yang vektornya paling mirip dengan query
        hits = util.semantic_search(query_vec, self.corpus_embeddings, top_k=top_k)[0]
        
        candidates = []
        q_tokens = self.clean(query)
        
        # --- TAHAP 2: FEATURE EXTRACTION (REAL-TIME) ---
        for hit in hits:
            doc_idx = hit['corpus_id']
            t_tokens = self.tokenized_corpus[doc_idx]
            
            # A. Hitung Overlap & Jaccard
            set_q = set(q_tokens)
            set_t = set(t_tokens)
            
            overlap, jaccard = 0.0, 0.0
            if len(set_q) > 0:
                # Overlap: Berapa % kata query yang muncul di teks?
                overlap = len(set_q.intersection(set_t)) / len(set_q)
                # Jaccard: Seberapa mirip himpunan katanya?
                jaccard = len(set_q.intersection(set_t)) / (len(set_q.union(set_t)) + 1e-9)
            
            # B. Hitung BM25 Score (Spesifik Query vs Dokumen ini)
            bm25_score = self.bm25.get_batch_scores(q_tokens, [doc_idx])[0]
            
            # Simpan data mentah + fitur
            candidates.append({
                'text': self.corpus[doc_idx],     # Teks asli untuk ditampilkan
                'sbert_sim': hit['score'],        # Fitur 1
                'bm25_score': bm25_score,         # Fitur 2
                'overlap_score': overlap,         # Fitur 3
                'jaccard_score': jaccard          # Fitur 4
            })
            
        return candidates