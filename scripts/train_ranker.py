import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve

# ==========================================
# 1. KONFIGURASI DINAMIS (Sesuai Struktur GitHub)
# ==========================================
# Mendapatkan lokasi script ini berada (folder 'scripts/')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Mendapatkan root folder proyek (naik satu level dari 'scripts/')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Path Input & Output
DATA_INPUT = os.path.join(PROJECT_ROOT, 'data', 'processed', 'Sampel_datatrain.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_OUTPUT = os.path.join(MODEL_DIR, 'xgboost_best_model.json')

# Konfigurasi Model
TARGET_RECALL = 0.85
FEATURES = ['sbert_sim', 'bm25_score', 'overlap_score', 'jaccard_score']
TARGET = 'label'

def setup_directories():
    """Memastikan folder output tersedia."""
    if not os.path.exists(MODEL_DIR):
        print(f"📁 Membuat folder model: {MODEL_DIR}")
        os.makedirs(MODEL_DIR)

def load_data(filepath):
    """Memuat dan membersihkan dataset."""
    print(f"📂 Membaca data dari: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ FATAL ERROR: Dataset tidak ditemukan di {filepath}")
        print("   Pastikan file 'Sampel_datatrain.csv' sudah ada di folder 'data/processed/'")
        sys.exit(1)

    try:
        # Load data (handle delimiter otomatis)
        df = pd.read_csv(filepath, on_bad_lines='skip')
        if len(df.columns) <= 1:
            df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
            
        # Bersihkan nama kolom
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
        
        # Konversi numerik & Drop NaN
        print("   -> Membersihkan tipe data...")
        for col in FEATURES + [TARGET]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_clean = df.dropna(subset=FEATURES + [TARGET])
        print(f"✅ Data siap: {len(df_clean)} baris (Clean).")
        return df_clean
        
    except Exception as e:
        print(f"❌ Error membaca CSV: {e}")
        sys.exit(1)

def get_optimal_threshold(y_true, y_prob, target_recall):
    """Mencari threshold probabilitas untuk mencapai target recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Cari indeks di mana recall >= target_recall
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) > 0:
        idx = valid_indices[-1]
        threshold = thresholds[idx]
        return threshold, recall[idx], precision[idx]
    else:
        return 0.5, 0.0, 0.0

def main():
    setup_directories()
    
    # 1. Load Dataset
    df = load_data(DATA_INPUT)
    X = df[FEATURES]
    y = df[TARGET]

    # 2. Hitung Class Weight (Penyeimbang Data)
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    ratio = float(neg) / float(pos)
    print(f"⚖️ Rasio Kelas: {ratio:.2f} (Negatif: {neg}, Positif: {pos})")

    # 3. Split Data (Training & Validasi Internal)
    print("\n✂️ Membagi data (80% Train, 20% Val)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Training
    print("🔥 Memulai Training XGBoost...")
    # Deteksi GPU otomatis
    device_type = 'cuda' if os.system("nvidia-smi") == 0 else 'cpu'
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device=device_type, # Otomatis pilih GPU/CPU
        early_stopping_rounds=50,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    # 5. Evaluasi & Tuning Threshold
    print("\n🧠 Mengevaluasi & Mencari Threshold Optimal...")
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"⭐ ROC-AUC Score: {auc:.4f}")

    thresh, rec, prec = get_optimal_threshold(y_test, y_prob, TARGET_RECALL)
    
    print(f"\n🎯 HASIL OPTIMASI (Target Recall > {TARGET_RECALL*100}%):")
    print(f"   -> Threshold  : {thresh:.4f}")
    print(f"   -> Recall     : {rec:.4f}")
    print(f"   -> Precision  : {prec:.4f}")

    # Simpan Threshold ke file teks (agar bisa dibaca aplikasi nanti)
    thresh_file = os.path.join(MODEL_DIR, 'threshold.txt')
    with open(thresh_file, 'w') as f:
        f.write(str(thresh))
    print(f"   -> Nilai threshold disimpan di: {thresh_file}")

    # 6. Simpan Model
    model.save_model(MODEL_OUTPUT)
    print(f"\n💾 Model Tersimpan: {MODEL_OUTPUT}")
    
    # 7. Plot Feature Importance (Opsional, akan muncul popup window)
    try:
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({'Fitur': FEATURES, 'Importance': importance}).sort_values('Importance', ascending=False)
        print("\n🏆 Feature Importance:")
        print(feat_imp)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Importance', y='Fitur', data=feat_imp, palette='viridis')
        plt.title('Kontribusi Fitur XGBoost')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png')) # Simpan gambar juga
        plt.show()
    except Exception as e:
        print(f"⚠️ Gagal menampilkan plot: {e}")

if __name__ == "__main__":
    main()