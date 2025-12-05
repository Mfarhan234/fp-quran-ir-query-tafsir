# app/streamlit_app.py
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.relevance_model_loader import (
    AVAILABLE_MODELS,
    load_model_by_name,
    load_best_model,
    score_query_against_corpus,
)
from src.data.load_corpus import load_tafsir_corpus
# kalau nanti mau pakai pipeline rewrite query:
# from src.query.query_pipeline import build_final_query
# from sentence_transformers import SentenceTransformer

# ==========================
# CACHE RESOURCE
# ==========================

@st.cache_resource
def get_corpus() -> pd.DataFrame:
    return load_tafsir_corpus()

@st.cache_resource
def get_model(model_name: str):
    if model_name == "__BEST__":
        model, scaler, feat_names, use_scaled, best_name = load_best_model()
        return model, scaler, feat_names, use_scaled, best_name
    else:
        model, scaler, feat_names, use_scaled = load_model_by_name(model_name)
        return model, scaler, feat_names, use_scaled, model_name

# (opsional) kalau mau pakai embed_model untuk pipeline online:
# @st.cache_resource
# def get_embed_model():
#     return SentenceTransformer("intfloat/multilingual-e5-base")


# ==========================
# MAIN APP
# ==========================

def main():
    st.set_page_config(page_title="FP Qur'an IR - Tafsir", layout="wide")
    st.title("Pencarian Tafsir Ayat Al-Qur'an")

    # --- Sidebar: pilih model & top-k ---
    st.sidebar.header("Pengaturan Pencarian")

    model_options = ["[BEST OVERALL]"] + AVAILABLE_MODELS
    model_label = st.sidebar.selectbox("Pilih model relevansi", model_options)

    if model_label == "[BEST OVERALL]":
        model_key = "__BEST__"
    else:
        model_key = model_label

    top_k = st.sidebar.slider("Tampilkan Top-K ayat teratas", min_value=3, max_value=20, value=10, step=1)

    # Kalau mau aktifkan rewrite:
    # use_rewrite = st.sidebar.checkbox("Gunakan pipeline rewrite query (DeepSeek)", value=False)

    # --- Input query pengguna ---
    st.subheader("Tulis pertanyaan / curhatanmu")
    user_query = st.text_area(
        "Contoh: Saya sering lalai shalat karena kerja, bagaimana pandangan Al-Qur'an?",
        height=140,
    )

    if st.button("Cari Tafsir") and user_query.strip():
        df_corpus = get_corpus()
        with st.spinner("Menghitung relevansi..."):
            # 1. Query final (kalau mau pakai pipeline online, bisa ganti di sini)
            query_final = user_query.strip()

            # Kalau mau pakai rewrite:
            # if use_rewrite:
            #     embed_model = get_embed_model()
            #     from src.query.query_pipeline import build_final_query
            #     out = build_final_query(query_final, embed_model)
            #     query_final = out["final_query"]

            # 2. Load model
            model, scaler, feat_names, use_scaled, model_name = get_model(model_key)

            # 3. Hitung skor relevansi
            scores = score_query_against_corpus(
                query_text=query_final,
                df_corpus=df_corpus,
                model=model,
                scaler=scaler,
                feature_names=feat_names,
                use_scaled=use_scaled,
            )

            df_show = df_corpus.copy()
            df_show["score"] = scores
            df_show = df_show.sort_values("score", ascending=False).head(top_k)

        # --- Info ringkas ---
        st.markdown(
            f"**Model:** {model_name}  •  "
            f"Top-K: {top_k}  •  "
            f"Query akhir: _{query_final}_"
        )
        st.markdown("---")

        # --- Tampilkan hasil satu per satu ---
        for idx, row in df_show.iterrows():
            surah = row["surah"]
            ayah = row["ayah"]
            arab = row["arabic_text"]
            indo = row["indonesian_translation"]
            tafsir = row["tafsir"]
            score = row["score"]

            st.markdown(f"### {surah} : {ayah}  — skor relevansi: {score:.3f}")
            st.markdown(f"**Teks Arab:**")
            st.markdown(f"<div style='font-size: 22px; direction: rtl; text-align: right;'>{arab}</div>", unsafe_allow_html=True)
            st.markdown(f"**Terjemahan Indonesia:**")
            st.write(indo)
            st.markdown(f"**Tafsir (ringkas):**")
            st.write(tafsir)
            st.markdown("---")
    else:
        st.info("Masukkan pertanyaan di atas lalu klik **Cari Tafsir**.")


if __name__ == "__main__":
    main()
