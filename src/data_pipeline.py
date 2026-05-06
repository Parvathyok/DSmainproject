import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Telco-Customer-Churn.csv')
    return pd.read_csv(data_path)

def preprocess_data(df):
    # Drop customer ID
    df = df.drop(columns=['customerID'])
    
    # Handle TotalCharges missing values (blank spaces in the raw csv)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Save the label encoders to use them during inference
    label_encoders = {}
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    # Features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    return X, y, label_encoders

def get_train_test_split():
    df = load_data()
    X, y, label_encoders = preprocess_data(df)
    
    # Scale continuous features
    scaler = StandardScaler()
    continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler, label_encoders

if __name__ == "__main__":
    import pickle
    X_train, X_test, y_train, y_test, scaler, encoders = get_train_test_split()
    print(f"Data preprocessed. Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'transformers')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
        
    print(f"Processed data saved to {processed_dir}")
    print(f"Transformers saved to {models_dir}")
