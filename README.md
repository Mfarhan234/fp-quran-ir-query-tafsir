# Al- Qur'an Information Retrieval System

Sistem pencarian tafsir Al-Qur'an berbasis **Two-Stage Retrieval** yang mengintegrasikan SBERT dan Machine Learning untuk menjembatani *lexical gap* antara bahasa sehari-hari dengan teks formal tafsir.

![1766112933153](image/README/1766112933153.png)

## Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Struktur Proyek](#struktur-proyek)
- [Pipeline Penelitian](#pipeline-penelitian)
- [Performa Model](#performa-model)
- [Dataset](#dataset)
- [Teknologi](#teknologi)
- [Kontributor](#kontributor)
- [Lisensi](#lisensi)

## Tentang Proyek

Proyek tugas akhir ini mengembangkan sistem Information Retrieval (IR) untuk mencari tafsir Al-Qur'an menggunakan pendekatan hibrida yang menggabungkan kekuatan pencarian semantik (SBERT) dan peringkat berbasis machine learning. Sistem ini dirancang untuk memahami pertanyaan dalam bahasa natural dan mengembalikan ayat beserta tafsirannya yang paling relevan.

## Fitur Utama

- **Two-Stage Retrieval Architecture**: Kombinasi SBERT retrieval dan ML re-ranking
- **Hybrid Features**: Integrasi fitur semantik (SBERT) dan leksikal (BM25, Jaccard, Overlap)
- **Multiple Models**: Mendukung 5 model ranking (XGBoost, LightGBM, Random Forest, Logistic Regression, SVM)
- **Streamlit Web App**: Interface interaktif untuk pencarian real-time
- **High Accuracy**: nDCG@3 mencapai 0.9648 dengan model XGBoost
- **Indonesian Language Support**: Optimized untuk query berbahasa Indonesia

## Arsitektur Sistem

Sistem menggunakan **Two-Stage Retrieval Architecture**:

```
Query User
    ↓
[Stage 1: SBERT Retrieval]
    → Mencari top-50 kandidat dari 6,246 ayat
    ↓
[Feature Extraction]
    → SBERT Similarity
    → BM25 Score
    → Jaccard Similarity
    → Overlap Coefficient
    ↓
[Stage 2: ML Re-Ranking]
    → XGBoost/LightGBM/RF/LR/SVM
    ↓
Top-5 Hasil Terurut
```

## Instalasi

### Prasyarat

- Python 3.8+
- pip atau conda

### Langkah Instalasi

```bash
# Clone repository
git clone https://github.com/yourusername/fp-quran-ir-query-tafsir.git
cd fp-quran-ir-query-tafsir

# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords (otomatis saat pertama kali run)
```

## Cara Penggunaan

### Menjalankan Web Application

```bash
streamlit run app/streamlit_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

### Melatih Ulang Model

Jalankan notebook secara berurutan:

```bash
# 1. Eksplorasi data
notebooks/01_eda_dataset_alquran.ipynb

# 2. Generate synthetic queries
notebooks/02_create_query_final.ipynb

# 3. Hard negative mining
notebooks/03_hard_negative_labeling.ipynb

# 4. Feature engineering
notebooks/04_feature_extraction.ipynb

# 5. Train models
notebooks/05_Train_ensemble.ipynb
notebooks/06_Train_single_model.ipynb

# 6. Evaluasi
notebooks/07_Evaluasi_Ensamble&Single_Model.ipynb
```

## Struktur Proyek

```
fp-quran-ir-query-tafsir/
├── app/
│   ├── streamlit_app.py          # Main application
│   └── config_example.yaml        # Configuration template
├── data/
│   ├── raw/                       # Raw Al-Quran dataset
│   └── processed/                 # Processed datasets
│       ├── tafsir_clean.csv       # Clean tafsir data
│       └── dataset_training_*.csv # Training data
├── models/
│   ├── sbert_finetuned_quran/     # Fine-tuned SBERT
│   ├── corpus_embeddings.pt       # Pre-computed embeddings
│   ├── xgboost_best_model.json    # XGBoost model
│   ├── lightgbm.pkl               # LightGBM model
│   ├── randomforest.pkl           # Random Forest model
│   ├── logisticregression.pkl     # Logistic Regression
│   └── svm.pkl                    # SVM model
├── notebooks/                     # Jupyter notebooks
└── requirements.txt               # Python dependencies
```

## Pipeline Penelitian

### 1. Konstruksi Dataset & Auto-Labeling

Karena keterbatasan data label relevansi, sistem menggunakan **weak supervision**:

- **Data Dasar**: Dataset Al-Qur'an Indonesia (6,236 ayat dengan tafsir)
- **Synthetic Query Generation**: Menggunakan LLM (DeepSeek V3) untuk generate ~42,000 kueri
- **Hard Negative Mining**: Strategi *Skip Top-5* untuk mining 3 hard negatives per query
- **Output**: ~170,000 pasangan query-tafsir dengan label

### 2. Ekstraksi Fitur Hibrida

Setiap pasangan kueri-tafsir dikonversi menjadi **4 fitur numerik**:

| Fitur               | Tipe     | Deskripsi                         |
| ------------------- | -------- | --------------------------------- |
| SBERT Similarity    | Semantik | Cosine similarity dari embeddings |
| BM25 Score          | Leksikal | Probabilistic ranking function    |
| Jaccard Similarity  | Leksikal | Set intersection/union ratio      |
| Overlap Coefficient | Leksikal | Normalized word overlap           |

### 3. Pelatihan Model (Learning to Rank)

**Pendekatan**: Pointwise Learning to Rank (klasifikasi biner)

**Model yang Diuji**:

- Ensemble: XGBoost, LightGBM, Random Forest
- Single: SVM, Logistic Regression

**Model Terbaik**: XGBoost dengan hyperparameter tuning

### 4. Deployment

- **Framework**: Streamlit
- **Inference**: Real-time two-stage retrieval
- **Latency**: ~1-2 detik per query

## Performa Model

### Hasil Evaluasi (Test Set)

| Model               | nDCG@3           | MRR              | MAP              | Precision@5      |
| ------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| **XGBoost**   | **0.9648** | **0.9531** | **0.9234** | **0.9150** |
| LightGBM            | 0.9612           | 0.9498           | 0.9201           | 0.9120           |
| Random Forest       | 0.9587           | 0.9467           | 0.9178           | 0.9080           |
| SVM                 | 0.9497           | 0.9327           | 0.9045           | 0.8950           |
| Logistic Regression | 0.9456           | 0.9289           | 0.8998           | 0.8920           |

### Ablation Study

Peningkatan performa dengan integrasi fitur hibrida:

- **MAP improvement**: +21.73% vs BM25-only baseline
- **nDCG@3 improvement**: +15.42% vs SBERT-only baseline

## Dataset

### Sumber Data

- **Dataset**: Aban R. Al-Quran Indonesia Dataset. Kaggle. 2024. Available from: https://www.kaggle.com/datasets/ronnieaban/alquran
- **Jumlah Ayat**: 6,236 ayat dengan terjemahan dan tafsir
- **Bahasa**: Indonesia
- **Format**: CSV dengan kolom: surah, ayah, arabic_text, indonesian_translation, tafsir

### Data Sintetik

- **Synthetic Queries**: ~42,000 kueri (generated dengan DeepSeek V3)
- **Training Pairs**: ~170,000 pasangan query-tafsir (positive + hard negatives)
- **Split**: 80% train, 10% validation, 10% test

## Teknologi

### Core Libraries

- **Sentence Transformers**: Fine-tuned SBERT untuk semantic search
- **XGBoost**: Gradient boosting untuk ranking
- **LightGBM**: Alternatif fast gradient boosting
- **Streamlit**: Web application framework
- **Rank-BM25**: Implementation BM25 algorithm
- **PyTorch**: Deep learning backend

### Development Tools

- **Jupyter Notebook**: Eksperimen dan analisis
- **Pandas**: Data manipulation
- **Scikit-learn**: Model evaluation metrics
- **NLTK**: Text preprocessing

## Kontributor

Proyek ini dikembangkan oleh Kelompok 8 - Data Mining, Institut Teknologi Sepuluh Nopember (ITS):

1. **Muhammad Farhan** (5054241018)
2. **Muhammad Dayyan Ghazanfar Latief** (5054241036)
3. **Izzah Naufalia Adila** (5054241021)

## Acknowledgments

- Aban R. untuk dataset Al-Qur'an Indonesia di Kaggle
- Sentence Transformers library oleh UKPLab
- Inspirasi dari paper InPars (Bonifacio et al., 2022)

**Note**: Proyek ini adalah bagian dari Tugas Akhir di Teknik Informatika ITS untuk pengembangan sistem Information Retrieval pada domain Al-Qur'an berbahasa Indonesia.
