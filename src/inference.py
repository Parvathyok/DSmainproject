import os
import pickle
import pandas as pd

class ChurnPredictor:
    def __init__(self):
        transformers_dir = os.path.join(os.path.dirname(__file__), 'transformers')
        
        with open(os.path.join(transformers_dir, 'churn_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
            
        with open(os.path.join(transformers_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(os.path.join(transformers_dir, 'encoders.pkl'), 'rb') as f:
            self.encoders = pickle.load(f)
            
        self.continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    def predict(self, user_data: dict):
        """
        Expects a dictionary with keys matching the original dataframe 
        (minus customerID and Churn).
        """
        df = pd.DataFrame([user_data])
        
        # Numeric parsing
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
            
        # Transform categorical cols
        for col, le in self.encoders.items():
            if col in df.columns and col != 'Churn':
                df[col] = le.transform(df[col])
                
        # Scale continuous features
        df[self.continuous_cols] = self.scaler.transform(df[self.continuous_cols])
        
        # Predict probability
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        return {
            'churn_prediction': bool(prediction),
            'churn_probability': float(probability)
        }
