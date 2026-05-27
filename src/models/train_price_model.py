import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_dummy_price_data(weather_df):
    """
    Creates a synthetic price dataset that aligns with the weather data dates
    so we can test the pipeline before you download the real Agmarknet data.
    """
    dates = pd.to_datetime(weather_df['time'])
    np.random.seed(42)
    
    # Base price of Tomato (Rs per Quintal, approx Rs 2000 = Rs 20/kg)
    base_price = 2000
    
    # Random walk with some seasonal noise
    prices = [base_price]
    for _ in range(1, len(dates)):
        change = np.random.normal(0, 50)
        new_price = max(500, prices[-1] + change)
        prices.append(new_price)
        
    df = pd.DataFrame({
        'Date': dates.dt.strftime('%d-%b-%Y'),
        'Modal_Price': prices
    })
    
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/agmarknet_78_TN_dummy.csv", index=False)
    print("Dummy price data generated for testing.")
    return df

def prepare_data(weather_path, price_path):
    print("Loading datasets...")
    weather_df = pd.read_csv(weather_path)
    
    if not os.path.exists(price_path):
        print(f"Price data not found at {price_path}. Creating dummy data for testing...")
        price_df = create_dummy_price_data(weather_df)
    else:
        price_df = pd.read_csv(price_path)
        
    # Preprocessing
    weather_df['date'] = pd.to_datetime(weather_df['time'])
    
    # Agmarknet dates are usually like '01-Jan-2023'
    # We will try a few formats just in case
    try:
        price_df['date'] = pd.to_datetime(price_df['Date'], format='%d-%b-%Y')
    except:
        price_df['date'] = pd.to_datetime(price_df['Date'])
        
    # Merge datasets
    df = pd.merge(price_df, weather_df, on='date', how='inner')
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Feature Engineering
    # 1. Target Variable: Price in 7 days
    df['target_price_7d'] = df['Modal_Price'].shift(-7)
    
    # 2. Lag Features
    for i in [1, 3, 7]:
        df[f'price_lag_{i}'] = df['Modal_Price'].shift(i)
        
    # 3. Rolling Averages
    df['price_roll_mean_7'] = df['Modal_Price'].rolling(window=7).mean()
    df['temp_roll_mean_7'] = df['temperature_2m_max'].rolling(window=7).mean()
    df['precip_roll_sum_7'] = df['precipitation_sum'].rolling(window=7).sum()
    
    # 4. Date Features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    # Drop rows with NaN due to shifting/rolling
    df = df.dropna()
    
    return df

def train_model_for_crop(df, crop_name):
    print(f"\n--- Training Model for {crop_name} ---")
    print(f"Training on {len(df)} days of data...")
    
    features = [
        'Modal_Price', 'price_lag_1', 'price_lag_3', 'price_lag_7',
        'price_roll_mean_7', 'temperature_2m_max', 'temp_roll_mean_7', 
        'precipitation_sum', 'precip_roll_sum_7', 'relative_humidity_2m_mean',
        'day_of_year', 'month'
    ]
    target = 'target_price_7d'
    
    X = df[features]
    y = df[target]
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"MAE: Rs {mae:.2f} per Quintal")
    print(f"RMSE: Rs {rmse:.2f} per Quintal")
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/xgboost_price_model_{crop_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {model_path}")

if __name__ == "__main__":
    import glob
    
    weather_files = glob.glob("data/raw/weather_madurai_*.csv")
    if not weather_files:
        print("Please run fetch_weather.py first.")
        exit(1)
        
    weather_path = sorted(weather_files)[-1]
    
    crops = ["Tomato", "Onion", "Groundnut"]
    for crop in crops:
        price_path = f"data/raw/agmarknet_78_TN_{crop}.csv"
        if os.path.exists(price_path):
            df = prepare_data(weather_path, price_path)
            train_model_for_crop(df, crop)
        else:
            print(f"Missing price data for {crop}. Run filter_dataset.py first.")
