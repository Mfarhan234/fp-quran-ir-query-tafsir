
Berikut draft **README.md** yang bisa langsung kamu pakai/diedit di repo `fp-quran-ir-query-tafsir`.

---

# FP Qur'an IR – Pencarian Ayat & Tafsir Berbasis Machine Learning

Proyek ini membangun sistem pencarian ayat Al-Qur’an berbahasa Indonesia.

Pengguna bisa menulis pertanyaan / curhatan dalam bahasa natural, lalu sistem:

1. Mengubah input pengguna menjadi query yang lebih fokus (opsional: dengan LLM DeepSeek).
2. Menghitung skor relevansi antara query dan teks tafsir menggunakan model ML (single & ensemble).
3. Mengembalikan **Top-K ayat** beserta:
   * Nama surah & nomor ayat
   * Teks Arab
   * Terjemahan Indonesia
   * Tafsir (inti utama)

Sistem akhir disajikan dalam bentuk aplikasi  **Streamlit** .

---

## 1. Struktur Repo

Struktur utama (disederhanakan):

```text
fp-quran-ir-query-tafsir/
├─ app/
│  └─ streamlit_app.py           # UI utama Streamlit
├─ data/
│  ├─ raw/
│  │  └─ Al_Quran_Dataset.csv    # dataset dasar ayat (opsional)
│  └─ processed/
│     └─ tafsir_clean.csv        # korpus tafsir bersih (dipakai saat inference)
├─ experiments/
│  └─ notebooks/
│     ├─ 00_baseline_semantic_search.ipynb
│     ├─ 01_eda_dataset_alquran.ipynb
│     ├─ 02_create_sinthetic_query_llm.ipynb
│     ├─ 03_feature_engineering.ipynb
│     └─ 04_model-single-and-ensamble.ipynb
├─ models/
│  ├─ best_relevance_model.pkl   # model terbaik (ranking)
│  ├─ model_XGBoost.pkl          # model per algoritma (opsional)
│  ├─ model_LightGBM.pkl
│  ├─ model_LogisticRegression.pkl
│  ├─ model_RandomForest.pkl
│  ├─ model_SVM.pkl
│  └─ model_comparison_results.csv
├─ src/
│  ├─ data/
│  │  └─ load_corpus.py          # load tafsir_clean.csv
│  ├─ evaluation/
│  ├─ features/
│  │  └─ feature_builder.py      # pembuatan fitur query–tafsir
│  ├─ llm/
│  │  └─ deepseek_client.py      # wrapper DeepSeek API
│  ├─ models/
│  │  └─ relevance_model_loader.py  # load .pkl + scoring
│  └─ query/
│     └─ query_pipeline.py       # pipeline penangkapan makna query (online)
├─ .env                          # kunci API (TIDAK di-commit)
├─ .gitignore
├─ README.md
└─ requirements.txt
```

---

## 2. Persyaratan

* **Python** 3.11 atau 3.12 (disarankan 3.12.x)
* Pip (sudah termasuk di Python)
* (Opsional) Git & VS Code

Library utama (sudah tercantum di `requirements.txt`):

* `streamlit`
* `pandas`, `numpy`
* `scikit-learn`
* `xgboost`
* `lightgbm`
* `sentence-transformers` (untuk embedding E5)
* dll.

---

## 3. Setup Environment (Virtual Env)

Dari root repo `fp-quran-ir-query-tafsir`:

```bash
# 1. Buat virtual environment (contoh Python 3.12)
py -3.12 -m venv .venv

# 2. Aktifkan venv (Git Bash / bash)
source .venv/Scripts/activate

# 3. Install dependencies
pip install -r requirements.txt
```

Jika menggunakan CMD:

```cmd
.\.venv\Scripts\activate.bat
pip install -r requirements.txt

Menggunakan Streamlit:
streamlit run app/streamlit_app.py
```

---

## 4. Menyiapkan Data & Model

### 4.1. Korpus Tafsir

File yang digunakan saat inference:

```text
data/processed/tafsir_clean.csv
```

Format minimal (nama kolom):

* `surah` – nama surah
* `ayah` – nomor ayat
* `arabic_text` – teks Arab ayat
* `indonesian_translation` – terjemahan Indonesia
* `tafsir` – teks tafsir (basis pencarian)

Jika nama kolommu berbeda, sesuaikan mapping di:

```python
src/data/load_corpus.py
```

### 4.2. Model Relevansi (.pkl)

Model ranking yang sudah dilatih (biasanya dari Kaggle) diletakkan di:

```text
models/
  best_relevance_model.pkl
  model_XGBoost.pkl
  model_LightGBM.pkl
  model_LogisticRegression.pkl
  model_RandomForest.pkl
  model_SVM.pkl
  model_comparison_results.csv
```

* `best_relevance_model.pkl` → dipakai saat user memilih **[BEST OVERALL]** di UI.
* `model_*.pkl` → bisa dipilih manual di sidebar Streamlit (XGBoost saja, LightGBM saja, dst).

Isi setiap `.pkl` kurang lebih:

```python
{
  "model": <estimator terlatih>,
  "scaler": <StandardScaler atau None>,
  "feature_names": [list_nama_fitur],
  "use_scaled": True/False,
  "model_name": "XGBoost" / dll
}
```

File ini di-load oleh `src/models/relevance_model_loader.py`.

---

## 5. Menjalankan Aplikasi Streamlit

Pastikan:

* Virtual env **aktif**
* Dependency sudah ter-install
* `data/processed/tafsir_clean.csv` dan folder `models/` sudah terisi

Lalu jalankan:

```bash
streamlit run app/streamlit_app.py
```

Browser akan otomatis terbuka (default: `http://localhost:8501`).

### 5.1. Cara Pakai UI

1. **Sidebar**
   * Pilih  **model** :
     * `[BEST OVERALL]` → `best_relevance_model.pkl`
     * atau salah satu: `XGBoost`, `LightGBM`, `LogisticRegression`, `RandomForest`, `SVM`
   * Atur **Top-K** (misal 10) untuk banyaknya ayat yang ditampilkan.
   * (Opsional) toggle penggunaan **pipeline rewrite query** jika diaktifkan.
2. **Input utama**
   * Tulis pertanyaan/cerita dalam Bahasa Indonesia, contoh:
     > “Saya sering lalai shalat karena kerja, bagaimana pandangan Al-Qur’an?”
     >
3. Klik **"Cari Tafsir"**
4. Aplikasi akan menampilkan daftar  **Top-K ayat paling relevan** :
   * `Surah : Ayah — skor relevansi`
   * Teks Arab (rtl)
   * Terjemahan Indonesia
   * Tafsir (inti penjelasan)

---

## 6. Pipeline Penangkapan Makna Query (Online)

Untuk sesi online (saat user bertanya):

1. **Raw input user**

   Kalimat panjang / curhatan lengkap.
2. (Opsional) **Compress** dengan embedding + K-Means

   * Menggunakan `SentenceTransformer("intfloat/multilingual-e5-base")`
   * Dipecah per kalimat, di-cluster → diambil kalimat paling representatif.
3. (Opsional) **Rewrite dengan LLM (DeepSeek)**

   * Fungsi di `src/query/query_pipeline.py` memanggil `DeepSeekClient`
   * Prompt meminta LLM merangkum serta mengekspresikan maksud pertanyaan secara jelas.
4. **Final query** → masuk ke modul fitur + model relevansi

   * Dibentuk fitur (TF-IDF, embedding, similarity, dll.) melalui

     `src/features/feature_builder.py`
   * Distandardisasi (jika perlu) dengan `scaler`.
   * Diberi skor relevansi oleh model (XGBoost, LightGBM, dsb.)
   * Top-K ayat dipilih dan ditampilkan.

> **Catatan:** Penggunaan DeepSeek membutuhkan API key yang disimpan di `.env`.

---

## 7. Konfigurasi Kunci API (.env)

Untuk fitur LLM (DeepSeek):

Buat file `.env` di root repo:

```env
# JANGAN di-commit ke Git
DEEPSEEK_API_KEY=sk-ISI_DENGAN_API_KEY_ANDA
```

* File `.env` sudah di-ignore di `.gitignore`.
* Key di-load oleh `src/llm/deepseek_client.py`.

Jika tidak ingin menggunakan LLM untuk sementara, pipeline online bisa dimatikan atau di-bypass (menggunakan query mentah langsung).

---

## 8. Retraining Model (Opsional)

Notebook untuk training ulang model:

* `experiments/notebooks/03_feature_engineering.ipynb`

  → menyiapkan `query_tafsir_features.csv` (feature matrix + label).
* `experiments/notebooks/04_model-single-and-ensamble.ipynb`

  → latihan dan evaluasi model:

  * Logistic Regression, SVM, RandomForest
  * XGBoost, LightGBM (ensemble)
  * Menyimpan `.pkl` dan `model_comparison_results.csv`.

Setelah retraining:

1. Salin `.pkl` baru ke folder `models/` di repo lokal.
2. Commit dan push ke GitHub (jika ukuran masih aman).

---

## 9. Catatan Pengembangan

* **Bahasa kerja** : Bahasa Indonesia, terutama di komentar dan README.
* Kode dirapikan dalam package `src/` supaya mudah di-import dari notebook dan Streamlit.
* Tujuan akhir: memfasilitasi user awam untuk:
  * Menanyakan persoalan kehidupan sehari-hari,
  * Mendapatkan ayat + tafsir yang relevan,
  * Dengan pipeline yang bisa dijelaskan secara ilmiah di laporan FP (single vs ensemble model, efek auto-labeling, dsb.).

---

Silakan kamu edit bagian-bagian yang terlalu formal / kurang sesuai gaya kamu (misalnya nama notebook, deskripsi FP, dsb.). Kalau mau, kita juga bisa buat versi README singkat khusus untuk “calon dosen/penguji” dan versi lengkap untuk tim dev.
