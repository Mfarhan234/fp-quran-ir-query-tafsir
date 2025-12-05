# src/query/query_pipeline.py

from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import re
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from src.llm.deepseek_client import DeepSeekClient


# Util: split & filter kalimat

SENT_MIN_CHARS = 10  # kalimat terlalu pendek akan dibuang


def split_into_sentences(text: str) -> List[str]:
    """Pisah teks panjang menjadi list kalimat sederhana."""
    text = text.replace("\n", " ")
    parts = re.split(r"[\.!?]+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def is_greeting_or_noise(sent: str) -> bool:
    """Heuristik sederhana untuk buang sapaan/noise."""
    s = sent.lower()
    if len(s) < SENT_MIN_CHARS:
        return True

    patterns = [
        "assalamualaikum",
        "assalamu'alaikum",
        "ass wr wb",
        "halo",
        "hai",
        "hi ",
        "mau curhat",
        "curhat dikit",
        "selamat pagi",
        "selamat siang",
        "selamat malam",
    ]
    return any(p in s for p in patterns)


def filter_sentences(sentences: List[str]) -> List[str]:
    """Buang sapaan/noise, simpan kalimat yang lebih informatif."""
    cleaned: List[str] = []
    for s in sentences:
        if is_greeting_or_noise(s):
            continue
        cleaned.append(s.strip())

    if not cleaned:
        return sentences

    return cleaned


# Embedding + KMeans per query

def embed_sentences(
    sentences: List[str],
    embed_model: SentenceTransformer,
) -> np.ndarray:
    """Embed list kalimat jadi array numpy (n_sent, dim)."""
    if not sentences:
        return np.zeros((0, 0), dtype=np.float32)

    # kalau pakai e5, bisa ditambah prefix "query: ", tapi di sini kita pakai apa adanya
    texts = sentences
    emb = embed_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb


def pick_representative_sentences(
    sentences: List[str],
    emb: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> List[str]:
    """
    Cluster kalimat dan ambil 1 kalimat wakil per cluster
    (paling dekat ke centroid).
    """
    n = len(sentences)
    if n == 0:
        return []

    if n == 1 or emb.shape[0] == 1:
        return [sentences[0]]

    # k tidak boleh lebih besar dari jumlah kalimat
    k = min(n_clusters, n)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(emb)
    centroids = kmeans.cluster_centers_

    rep_indices: List[Tuple[int, int]] = []  # (idx_global, cluster_id)

    for cluster_id in range(k):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue

        vecs = emb[idxs]           # (m, dim)
        centroid = centroids[cluster_id]  # (dim,)
        sims = vecs @ centroid     # cosine karena sudah dinormalisasi

        best_local = int(np.argmax(sims))
        best_global_idx = int(idxs[best_local])
        rep_indices.append((best_global_idx, cluster_id))

    # urutkan wakil cluster sesuai urutan asli kalimat
    rep_indices_sorted = sorted(rep_indices, key=lambda x: x[0])
    reps = [sentences[i] for (i, _) in rep_indices_sorted]

    return reps


def compress_query_with_kmeans(
    user_input: str,
    embed_model: SentenceTransformer,
    max_clusters: int = 3,
) -> str:
    """
    Pipeline:
    1) split kalimat
    2) filter sapaan/noise
    3) embed per kalimat
    4) KMeans per query
    5) ambil kalimat wakil, gabungkan jadi compressed_text
    """
    raw = user_input.strip()
    if not raw:
        return ""

    sentences = split_into_sentences(raw)
    sentences = filter_sentences(sentences)

    emb = embed_sentences(sentences, embed_model)
    if emb.size == 0:
        return raw

    reps = pick_representative_sentences(
        sentences,
        emb,
        n_clusters=max_clusters,
    )
    compressed_text = " ".join(reps)

    return compressed_text.strip()


# DeepSeek: rewrite query

SYSTEM_PROMPT = """
Kamu adalah modul penulis ulang query untuk sistem pencarian ayat dan tafsir Al-Qur'an.
Tugasmu hanya merapikan pertanyaan, bukan menjawabnya.
""".strip()

USER_PROMPT_TEMPLATE = """
Berikut ini ringkasan curhatan pengguna yang sudah dipadatkan:

"{compressed_text}"

Instruksi:
1. Tulis SATU kalimat pertanyaan yang paling mewakili inti yang ingin ditanyakan pengguna.
2. Jangan menjawab pertanyaannya.
3. Jangan menambahkan topik baru yang tidak disebutkan pengguna.
4. Jangan menyebut nama surah atau nomor ayat tertentu.
5. Output hanya berupa SATU kalimat tanya dalam Bahasa Indonesia yang jelas dan sopan.
""".strip()


def rewrite_query_with_deepseek(
    compressed_text: str,
    embed_model: Optional[SentenceTransformer] = None,
    min_similarity: float = 0.6,
    deepseek_client: Optional[DeepSeekClient] = None,
) -> str:
    """
    Panggil DeepSeek untuk menulis ulang compressed_text jadi 1 kalimat tanya.
    Jika embed_model disediakan, cek kesamaan makna. Kalau terlalu beda, fallback.
    """
    compressed_text = compressed_text.strip()
    if not compressed_text:
        return ""

    if deepseek_client is None:
        deepseek_client = DeepSeekClient()  # baca config dari .env

    user_prompt = USER_PROMPT_TEMPLATE.format(compressed_text=compressed_text)

    llm_output = deepseek_client.chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=48,
        temperature=0.2,
    )

    query_rewrite = llm_output.strip()

    # optional: cek kesamaan makna dengan embedding
    if embed_model is not None and query_rewrite:
        emb = embed_model.encode(
            [compressed_text, query_rewrite],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        v1, v2 = emb[0], emb[1]
        sim = float(v1 @ v2)

        if sim < min_similarity:
            # terlalu jauh maknanya → fallback ke compressed_text
            return compressed_text

    return query_rewrite


# Wrapper utama untuk fase online

def build_final_query(
    user_input: str,
    embed_model: SentenceTransformer,
    max_clusters: int = 3,
    use_similarity_check: bool = True,
    deepseek_client: Optional[DeepSeekClient] = None,
) -> Dict[str, str]:
    """
    Wrapper sekali panggil:
    - input: user_input mentah
    - output: dict berisi query_raw, compressed_text, query_rewrite, final_query
    """
    query_raw = user_input.strip()

    compressed = compress_query_with_kmeans(
        user_input=query_raw,
        embed_model=embed_model,
        max_clusters=max_clusters,
    )

    query_rewrite = rewrite_query_with_deepseek(
        compressed_text=compressed,
        embed_model=embed_model if use_similarity_check else None,
        min_similarity=0.6,
        deepseek_client=deepseek_client,
    )

    # final_query bisa kamu ganti strategi (misal gabungan raw + rewrite)
    final_query = query_rewrite or compressed or query_raw

    return {
        "query_raw": query_raw,
        "compressed_text": compressed,
        "query_rewrite": query_rewrite,
        "final_query": final_query,
    }


# Contoh pemakaian manual (bisa kamu coba di VS Code)
if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-base"
    embed_model = SentenceTransformer(model_name)

    teks_user = """
    Assalamualaikum kak, mau curhat dikit.
    Akhir-akhir ini saya merasa mental saya drop, jadi sering banget ninggalin shalat.
    Saya takut Allah marah, tapi saya juga bingung harus mulai dari mana.
    Sebenarnya gimana sih pandangan Al-Qur'an tentang kondisi seperti ini dan apa yang sebaiknya saya lakukan?
    """

    result = build_final_query(teks_user, embed_model)
    print("RAW       :", result["query_raw"])
    print("COMPRESSED:", result["compressed_text"])
    print("REWRITE   :", result["query_rewrite"])
    print("FINAL     :", result["final_query"])
