import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def generate_disease_data():
    """
    Since structured public datasets for daily village-level pest outbreaks are 
    rare, we simulate realistic biological conditions based on agronomy.
    Fungal diseases (e.g., Leaf Blight) thrive in High Humidity + High Temp.
    Pest attacks (e.g., Whiteflies/Thrips) thrive in Low Humidity + High Temp.
    """
    print("Generating biological disease correlation dataset...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # Generate random weather features
    temp_max = np.random.uniform(22, 42, n_samples)
    humidity = np.random.uniform(30, 95, n_samples)
    rainfall = np.random.exponential(scale=5, size=n_samples) # mostly low rain, some spikes
    
    disease_labels = []
    
    for t, h, r in zip(temp_max, humidity, rainfall):
        # Biological logic mapping
        if h > 80 and t > 28 and r > 10:
            disease_labels.append("Leaf Blight (Fungal)")
        elif h > 75 and t > 25 and t < 30:
            disease_labels.append("Powdery Mildew")
        elif h < 50 and t > 32:
            disease_labels.append("Whitefly Attack (Pest)")
        elif h < 60 and t > 35:
            disease_labels.append("Thrips (Pest)")
        else:
            disease_labels.append("Healthy / Low Risk")
            
    df = pd.DataFrame({
        'temperature_2m_max': temp_max,
        'relative_humidity_2m_mean': humidity,
        'precipitation_sum': rainfall,
        'Disease_Risk': disease_labels
    })
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/disease_risk_synthetic.csv", index=False)
    return df

def train_classification_model(df):
    print("Training Disease Risk Classification Model...")
    
    features = ['temperature_2m_max', 'relative_humidity_2m_mean', 'precipitation_sum']
    X = df[features]
    y = df['Disease_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest is highly explainable for agricultural classification
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print("\nModel Evaluation (Test Set):")
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, preds))
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_disease_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Model saved successfully at: {model_path}")

if __name__ == "__main__":
    df = generate_disease_data()
    train_classification_model(df)
