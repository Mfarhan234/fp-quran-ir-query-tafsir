import xgboost as xgb
import pandas as pd
import os

class Ranker:
    def __init__(self):
        # Cari file model JSON di folder models/
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_best_model.json')
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model JSON tidak ditemukan di: {MODEL_PATH}")
            
        self.bst = xgb.Booster()
        self.bst.load_model(MODEL_PATH)
        
    def predict(self, candidates_list):
        if not candidates_list:
            return pd.DataFrame()
            
        df = pd.DataFrame(candidates_list)
        
        # Urutan Fitur HARUS SAMA dengan waktu training
        features = ['sbert_sim', 'bm25_score', 'overlap_score', 'jaccard_score']
        
        dtest = xgb.DMatrix(df[features])
        df['final_score'] = self.bst.predict(dtest)
        
        return df