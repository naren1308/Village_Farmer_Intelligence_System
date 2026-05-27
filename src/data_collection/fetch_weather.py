import requests
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_weather_data(lat, lon, start_date, end_date):
    """
    Fetches historical weather data using the Open-Meteo Historical Weather API.
    Open-Meteo is free for non-commercial use and doesn't require an API key,
    making it a great alternative to scraping the IMD website directly.
    """
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parameters for the API request
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "relative_humidity_2m_mean"],
        "timezone": "Asia/Kolkata"
    }
    
    print(f"Fetching weather data for Lat: {lat}, Lon: {lon} from {start_date} to {end_date}...")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "daily" in data:
            df = pd.DataFrame(data["daily"])
            print(f"Successfully fetched {len(df)} days of weather data.")
            return df
        else:
            print("No daily data found in the response.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Coordinates for Madurai, Tamil Nadu (as an example)
    lat = 9.9252
    lon = 78.1198
    
    # Let's fetch data for the last 1 year
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    df = fetch_weather_data(lat, lon, start_date, end_date)
    
    if not df.empty:
        os.makedirs("data/raw", exist_ok=True)
        filename = f"data/raw/weather_madurai_{start_date}_to_{end_date}.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
