import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
    
    # We will use Random Forest as a baseline.
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'transformers')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'churn_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def evaluate_model():
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'))
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'transformers', 'churn_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    print("Training model...")
    train_model()
    print("Evaluating model...")
    evaluate_model()
