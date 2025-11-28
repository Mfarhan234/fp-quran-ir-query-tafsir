"""
Streamlit Search UI for Quran Query-Tafsir Ranking.

This module provides a simple web interface to search and rank tafsir passages
based on user queries using trained ML models.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

# Import from src
from src.data_loader import TafsirDataLoader
from src.features import FeatureExtractor
from src.models import RankingModel


def load_sample_data() -> pd.DataFrame:
    """Load sample tafsir data for demonstration."""
    loader = TafsirDataLoader()
    return loader.create_sample_data(n_queries=10, n_docs_per_query=5)


@st.cache_resource
def initialize_models():
    """Initialize and cache models and feature extractors."""
    # Load sample data
    loader = TafsirDataLoader()
    data = loader.create_sample_data(n_queries=10, n_docs_per_query=5)
    
    # Initialize feature extractor (TF-IDF only for speed)
    feature_extractor = FeatureExtractor(use_tfidf=True, use_sbert=False)
    
    # Fit TF-IDF on all texts
    all_texts = data['query'].tolist() + data['tafsir_text'].tolist()
    feature_extractor.fit_tfidf(all_texts)
    
    # Prepare training data
    queries = data['query'].tolist()
    documents = data['tafsir_text'].tolist()
    labels = data['relevance'].values
    
    # Extract features
    X = feature_extractor.extract_features(queries, documents)
    
    # Train a simple model
    model = RankingModel(model_type='logistic_regression')
    model.fit(X, labels)
    
    return model, feature_extractor, data


def search_tafsir(
    query: str,
    model: RankingModel,
    feature_extractor: FeatureExtractor,
    corpus: pd.DataFrame,
    top_k: int = 5
) -> List[Tuple[str, float, int, int]]:
    """
    Search for relevant tafsir passages given a query.
    
    Args:
        query: User search query
        model: Trained ranking model
        feature_extractor: Fitted feature extractor
        corpus: DataFrame containing tafsir passages
        top_k: Number of results to return
        
    Returns:
        List of (tafsir_text, score, surah, ayah) tuples
    """
    # Get unique tafsir passages
    unique_tafsirs = corpus.drop_duplicates(subset=['tafsir_text'])
    
    # Create query-document pairs
    n_docs = len(unique_tafsirs)
    queries = [query] * n_docs
    documents = unique_tafsirs['tafsir_text'].tolist()
    
    # Extract features and predict scores
    features = feature_extractor.extract_features(queries, documents)
    scores = model.predict_scores(features)
    
    # Create results
    results = []
    for i, (_, row) in enumerate(unique_tafsirs.iterrows()):
        results.append((
            row['tafsir_text'],
            scores[i],
            row.get('surah', 0),
            row.get('ayah', 0)
        ))
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Quran Tafsir Search",
        page_icon="📖",
        layout="wide"
    )
    
    st.title("📖 Quran Query-Tafsir Search")
    st.markdown("""
    This is a demonstration search interface for finding relevant tafsir (Quran commentary) 
    passages based on your query. The system uses machine learning to rank results by relevance.
    """)
    
    # Initialize models
    with st.spinner("Loading models and data..."):
        model, feature_extractor, data = initialize_models()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    top_k = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=5)
    
    st.sidebar.header("📊 Model Info")
    st.sidebar.info(f"""
    - **Model**: Logistic Regression
    - **Features**: TF-IDF
    - **Corpus Size**: {len(data)} documents
    """)
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter your search query:",
            placeholder="e.g., Apa makna taqwa dalam Islam?"
        )
    
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Sample queries
    st.markdown("**Sample queries:**")
    sample_queries = [
        "Apa makna taqwa dalam Islam?",
        "Bagaimana hukum shalat fardhu?",
        "Apa hikmah puasa Ramadhan?",
        "Bagaimana cara bertaubat?"
    ]
    
    cols = st.columns(len(sample_queries))
    for i, sample in enumerate(sample_queries):
        if cols[i].button(sample, key=f"sample_{i}", use_container_width=True):
            query = sample
            search_button = True
    
    # Display results
    if search_button and query:
        st.markdown("---")
        st.subheader(f"🔎 Results for: \"{query}\"")
        
        with st.spinner("Searching..."):
            results = search_tafsir(query, model, feature_extractor, data, top_k)
        
        if results:
            for i, (text, score, surah, ayah) in enumerate(results, 1):
                with st.expander(f"**Result {i}** (Score: {score:.4f})", expanded=(i <= 3)):
                    st.markdown(f"**Text:** {text}")
                    st.caption(f"📍 Surah {surah}, Ayah {ayah}")
        else:
            st.warning("No results found.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Quran Query-Tafsir Ranking ML Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
