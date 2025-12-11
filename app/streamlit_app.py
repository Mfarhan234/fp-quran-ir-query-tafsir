import streamlit as st
import pandas as pd
import lightgbm as lgb
import joblib # Untuk load LGBM/RF/LR/SVM
import xgboost as xgb
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import string
import nltk
from nltk.corpus import stopwords
import torch
import os
import pickle
import numpy as np

# --- 1. KONFIGURASI UMUM & CSS ---
st.set_page_config(
    page_title="Quran AI Search (All Models)",
    page_icon="🕌",
    layout="wide"
)

# CSS untuk Tampilan Arab yang Rapi
st.markdown("""
<style>
    .arabic-text {
        font-family: 'Traditional Arabic', 'Scheherazade New', serif;
        font-size: 28px;
        direction: rtl;
        text-align: right;
        color: #006644;
        margin-bottom: 10px;
        line-height: 1.8;
    }
    .translation {
        font-style: italic;
        color: #F5F5DC;
        border-left: 3px solid #28a745;
        padding-left: 10px;
        margin-bottom: 15px;
    }
    .score-badge {
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- 2. KONFIGURASI FITUR & MODEL ---

# Urutan Fitur Model SANGAT KRUSIAL
FEATURES_ORDER = {
    # ORDE STANDAR (Digunakan oleh XGBoost, dan RF Custom kita)
    'XGBOOST':       ['sbert_sim', 'bm25_score', 'overlap_score', 'jaccard_score'],
    'RANDOM FOREST': ['sbert_sim', 'bm25_score', 'overlap_score', 'jaccard_score'],
    
    # ORDER NON-STANDAR (Ditemukan di file LGBM/LogReg/SVM teman Anda)
    'LIGHTGBM':      ['sbert_sim', 'overlap_score', 'jaccard_score', 'bm25_score'],
    'LOGISTIC REGRESSION': ['sbert_sim', 'overlap_score', 'jaccard_score', 'bm25_score'],
    'SVM':           ['sbert_sim', 'overlap_score', 'jaccard_score', 'bm25_score']
}

# --- 3. FUNGSI LOAD SYSTEM ---
@st.cache_resource
def load_system():
    # A. Setup Path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(root_dir, 'models')
    data_dir = os.path.join(root_dir, 'data', 'processed')
    
    # File Paths
    csv_path = os.path.join(data_dir, 'dataset_training_FULL_COMPLETE.csv')
    emb_path = os.path.join(models_dir, 'corpus_embeddings.pt')
    
    # B. Load Dataset & Metadata
    if not os.path.exists(csv_path):
        st.error("Dataset tidak ditemukan!")
        st.stop()
        
    df_full = pd.read_csv(csv_path)
    df_full.columns = df_full.columns.str.strip().str.lower()
    df_index = df_full.drop_duplicates(subset=['text']).copy()
    unique_tafsirs = df_index['text'].astype(str).tolist()
    
    metadata_map = {}
    for _, row in df_index.iterrows():
        key = str(row['text']).strip()
        metadata_map[key] = {
            'lokasi': row.get('ayat_asal', 'Lokasi ?'),
            'arabic': row.get('arabic', ''),
            'trans': row.get('translation', '')
        }

    # C. Load Embeddings (Otak Lama)
    if not os.path.exists(emb_path):
        st.error("Embeddings (.pt) hilang!")
        st.stop()
    corpus_embeddings = torch.load(emb_path, map_location=torch.device('cpu'))

    # D. Load Models
    
    # 1. SBERT
    sbert_path = os.path.join(models_dir, 'sbert_finetuned_quran')
    sbert = SentenceTransformer(sbert_path, device='cpu')
    
    # 2. XGBOOST
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(models_dir, 'xgboost_best_model.json'))
    
    # 3. LIGHTGBM
    lgbm_model = joblib.load(os.path.join(models_dir, 'lightgbm (2) (1).pkl'))
    
    # 4. RANDOM FOREST
    rf_model = joblib.load(os.path.join(models_dir, 'randomforest_custom.pkl'))
    
    # 5. LOGISTIC REGRESSION
    lr_model = joblib.load(os.path.join(models_dir, 'logisticregression (3).pkl'))
    
    # 6. SVM
    svm_model = joblib.load(os.path.join(models_dir, 'svm (3) (1).pkl'))

    # Gabungkan semua model ke dalam satu dictionary
    all_models = {
        'XGBOOST': xgb_model,
        'LIGHTGBM': lgbm_model,
        'RANDOM FOREST': rf_model,
        'LOGISTIC REGRESSION': lr_model,
        'SVM': svm_model
    }

    return sbert, all_models, corpus_embeddings, unique_tafsirs, metadata_map

# Load Resources
try:
    with st.spinner("Memuat 5 Engine AI & Database..."):
        sbert_model, ALL_MODELS, corpus_embeddings, unique_tafsirs, metadata_map = load_system()
except Exception as e:
    st.error(f"Gagal memuat sistem: {e}")
    st.stop()

# --- 4. BUILD BM25 ---
@st.cache_resource
def build_bm25(_texts):
    try:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('indonesian'))
    except:
        stop_words = set(['yang', 'dan', 'di'])

    def clean(t):
        t = str(t).lower().translate(str.maketrans('', '', string.punctuation))
        return [w for w in t.split() if w not in stop_words]

    tokens = [clean(t) for t in _texts]
    return BM25Okapi(tokens), clean

bm25, clean_func = build_bm25(unique_tafsirs)

# --- 5. ENGINE LOGIC ---
def search_engine(query, model_name, threshold):
    
    # 1. Retrieval & Feature Calculation (Stage 1)
    q_vec = sbert_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_vec, corpus_embeddings, top_k=50)[0]
    
    candidates = []
    q_toks = clean_func(query)
    
    for hit in hits:
        idx = hit['corpus_id']
        txt = unique_tafsirs[idx]
        t_toks = clean_func(txt)
        
        # Hitung 4 fitur wajib
        sq, st = set(q_toks), set(t_toks)
        ov = len(sq & st) / len(sq) if sq else 0
        jac = len(sq & st) / (len(sq | st) + 1e-9)
        bm25_s = bm25.get_batch_scores(q_toks, [idx])[0]
        
        candidates.append({
            'text': txt,
            'sbert_sim': hit['score'],
            'bm25_score': bm25_s,
            'overlap_score': ov,
            'jaccard_score': jac
        })
        
    if not candidates: return []
    
    # 2. Re-Ranking (Stage 2)
    model = ALL_MODELS[model_name]
    df_cand = pd.DataFrame(candidates)
    
    # Tentukan urutan fitur yang benar untuk model ini
    feature_order = FEATURES_ORDER[model_name]
    X_pred = df_cand[feature_order]

    # Prediksi
    try:
        if model_name == 'XGBOOST':
            dtest = xgb.DMatrix(X_pred)
            scores = model.predict(dtest)
            
        elif hasattr(model, 'predict_proba') and not isinstance(model, SVC): # RF, LGBM, LogReg
            # Ambil probabilitas kelas 1 (skor 0-1)
            scores = model.predict_proba(X_pred)[:, 1]
            
        elif model_name == 'SVM':
            # Gunakan decision_function untuk SVM (skor -inf s.d +inf)
            scores = model.decision_function(X_pred)
            # Karena skor bisa negatif, threshold di bawah 0.0
            if threshold < 0.0: threshold = 0.0 # override default threshold if too sensitive
            
        else: # Fallback untuk model regresi
            scores = model.predict(X_pred)
            
    except Exception as e:
        st.warning(f"Error prediksi ({model_name}): {e}")
        return []
        
    df_cand['final_score'] = scores
    
    # 3. Filter & Sort
    # Khusus SVM/LogReg, jika threshold 0.0, kita tetap menampilkan hasil di atas 0.0
    results = df_cand[df_cand['final_score'] > threshold].sort_values('final_score', ascending=False)
    return results

# --- 6. USER INTERFACE ---

st.title("🕌 Quran Re-Ranking Engine (5 Model)")
st.markdown("Aplikasi ini menggunakan arsitektur **Two-Stage Retrieval** untuk mencari tafsir.")
st.write("Tahap 1: SBERT mencari 50 kandidat. Tahap 2: Model pilihan Anda (XGBoost/LightGBM/RF/LR/SVM) memberi nilai akhir.")


# Input Query
query = st.text_input("", placeholder="Masukkan pertanyaan Anda...")

# Sidebar
with st.sidebar:
    st.header("⚙️ Engine & Sensitivitas")
    model_choice = st.radio("Pilih Model Ranking:", list(ALL_MODELS.keys()), index=1) # Default ke LightGBM
    
    # Default threshold berdasarkan model type
    if model_choice in ['XGBOOST', 'LIGHTGBM', 'RANDOM FOREST']:
        default_thresh = 0.20
    else: # LR, SVM (Skor lebih kecil, perlu threshold rendah)
        default_thresh = 0.01 
        
    threshold_val = st.slider("Threshold Skor Relevansi:", -1.0, 1.0, default_thresh, 0.01, 
                              help="Nilai minimum skor prediksi agar ditampilkan. (Coba 0.01 untuk model Linear/SVM)")
    
    st.info(f"Model Aktif: **{model_choice}**")

if query:
    with st.spinner(f"Mencari dan me-ranking menggunakan {model_choice}..."):
        df_results = search_engine(query, model_choice, threshold_val)
    
    if not df_results.empty:
        st.success(f"Ditemukan {len(df_results)} hasil relevan menggunakan {model_choice}.")
        
        for i, row in df_results.head(5).iterrows():
            txt = row['text']
            score = row['final_score']
            
            # Ambil Metadata
            info = metadata_map.get(str(txt).strip())
            
            loc = info.get('lokasi', 'Lokasi ?')
            arab = info.get('arabic', '')
            trans = info.get('trans', '')
            
            # Tentukan warna badge
            score_color = "#28a745" if score > 0.5 else "#ffc107" if score > 0.0 else "#dc3545"
            
            # Tampilan Card
            with st.expander(f"{loc} (Skor: {score:.4f})", expanded=(i==0)):
                st.markdown(f"<span class='score-badge' style='background-color: {score_color}; color: white;'>Score {model_choice}: {score:.4f}</span>", unsafe_allow_html=True)
                st.markdown("---")
                
                if arab: st.markdown(f"<div class='arabic-text'>{arab}</div>", unsafe_allow_html=True)
                if trans: st.markdown(f"<div class='translation'>{trans}</div>", unsafe_allow_html=True)
                
                st.markdown("**Tafsir:**")
                st.write(txt)
    else:
        st.warning(f"Tidak ada hasil yang memenuhi threshold {threshold_val} menggunakan {model_choice}.")