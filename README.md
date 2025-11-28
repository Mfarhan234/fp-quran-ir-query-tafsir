# Quran Query-Tafsir Ranking

FP ML & DM kelompok 8 (Al-Quran)

A machine learning project for ranking Quran tafsir (commentary) passages based on user queries. This project implements information retrieval techniques using TF-IDF and Sentence-BERT features, combined with various ranking models.

## 📁 Project Structure

```
fp-quran-ir-query-tafsir/
├── data/               # Tafsir datasets (CSV/JSON)
├── notebooks/          # Jupyter notebooks for exploration
│   └── tutorial.ipynb  # Complete tutorial notebook
├── src/                # Source code modules
│   ├── __init__.py     # Package initialization
│   ├── data_loader.py  # Data loading utilities
│   ├── features.py     # Feature extraction (TF-IDF, SBERT)
│   ├── models.py       # Ranking models (LR, SVM, XGBoost)
│   └── evaluation.py   # Evaluation metrics (MAP, nDCG, MRR)
├── app/                # Streamlit web application
│   ├── __init__.py
│   └── streamlit_app.py # Search UI
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mfarhan234/fp-quran-ir-query-tafsir.git
cd fp-quran-ir-query-tafsir
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

### Using the Tutorial Notebook

```bash
jupyter notebook notebooks/tutorial.ipynb
```

## 📊 Features

### Data Loading
- Load tafsir data from CSV or JSON files
- Create sample data for demonstration
- Train/test split by query

### Feature Extraction
- **TF-IDF**: Term frequency-inverse document frequency vectorization
- **SBERT**: Sentence-BERT embeddings for semantic similarity
- Configurable feature combinations

### Ranking Models
- **Logistic Regression**: Fast and interpretable baseline
- **SVM**: Support Vector Machine with RBF kernel
- **XGBoost**: Gradient boosting for improved performance
- **LightGBM**: Light Gradient Boosting Machine

### Evaluation Metrics
- **MAP**: Mean Average Precision
- **nDCG**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Recall@K**: Recall at K documents
- **Precision@K**: Precision at K documents

## 💻 Usage Example

```python
from src.data_loader import TafsirDataLoader
from src.features import FeatureExtractor
from src.models import RankingModel
from src.evaluation import RankingMetrics, print_metrics

# 1. Load data
loader = TafsirDataLoader()
data = loader.create_sample_data()
train_data, test_data = loader.train_test_split()

# 2. Extract features
feature_extractor = FeatureExtractor(use_tfidf=True, use_sbert=False)
all_texts = data['query'].tolist() + data['tafsir_text'].tolist()
feature_extractor.fit_tfidf(all_texts)

X_train = feature_extractor.extract_features(
    train_data['query'].tolist(),
    train_data['tafsir_text'].tolist()
)
y_train = train_data['relevance'].values

# 3. Train model
model = RankingModel(model_type='logistic_regression')
model.fit(X_train, y_train)

# 4. Evaluate
metrics = RankingMetrics()
results = metrics.evaluate_model(model, feature_extractor, test_data)
print_metrics(results)
```

## 📋 Requirements

- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- xgboost>=2.0.0
- lightgbm>=4.0.0
- sentence-transformers>=2.2.0
- streamlit>=1.28.0
- joblib>=1.3.0

## 📖 Data Format

The expected data format is a CSV or JSON file with the following columns:
- `query`: The search query or question
- `tafsir_text`: The tafsir explanation text
- `relevance`: Relevance score (0 or 1)
- `surah` (optional): Surah number
- `ayah` (optional): Ayah number

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.
