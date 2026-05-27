import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import glob

st.set_page_config(page_title="VFIS | Tamil Nadu", page_icon="🌾", layout="wide")

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    h1, h2, h3 {color: #2E7D32;}
    .alert-box {padding: 15px; margin-bottom: 20px; border: 1px solid transparent; border-radius: 4px;}
    .alert-warning {color: #856404; background-color: #fff3cd; border-color: #ffeeba;}
    .alert-danger {color: #721c24; background-color: #f8d7da; border-color: #f5c6cb;}
    .alert-success {color: #155724; background-color: #d4edda; border-color: #c3e6cb;}
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Village Farmer Intelligence System (VFIS)")
st.markdown("**Empowering Tamil Nadu Farmers with Data-Driven Decisions**")

# Get the root directory (one level up from app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load Models ---
@st.cache_resource
def load_models(crop_name):
    xgb_model = None
    rf_model = None
    xgb_path = os.path.join(BASE_DIR, "models", f"xgboost_price_model_{crop_name}.pkl")
    rf_path = os.path.join(BASE_DIR, "models", "rf_disease_model.pkl")
    
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    return xgb_model, rf_model

# --- Live Data Processing ---
@st.cache_data
def get_live_data(crop_name):
    weather_dir = os.path.join(BASE_DIR, "data", "raw")
    weather_files = glob.glob(os.path.join(weather_dir, "weather_madurai_*.csv"))
    price_path = os.path.join(BASE_DIR, "data", "raw", f"agmarknet_78_TN_{crop_name}.csv")
    
    if not weather_files or not os.path.exists(price_path):
        return None
        
    weather_path = sorted(weather_files)[-1]
    
    weather_df = pd.read_csv(weather_path)
    price_df = pd.read_csv(price_path)
    
    weather_df['date'] = pd.to_datetime(weather_df['time'])
    try:
        price_df['date'] = pd.to_datetime(price_df['Date'], format='%d-%b-%Y')
    except:
        price_df['date'] = pd.to_datetime(price_df['Date'])
        
    df = pd.merge(price_df, weather_df, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate live features for XGBoost
    for i in [1, 3, 7]:
        df[f'price_lag_{i}'] = df['Modal_Price'].shift(i)
        
    df['price_roll_mean_7'] = df['Modal_Price'].rolling(window=7).mean()
    df['temp_roll_mean_7'] = df['temperature_2m_max'].rolling(window=7).mean()
    df['precip_roll_sum_7'] = df['precipitation_sum'].rolling(window=7).sum()
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    return df.dropna().reset_index(drop=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Farmer Settings")
    village = st.selectbox("Select District", ["Madurai"])
    crop = st.selectbox("Select Crop", ["Tomato", "Onion", "Groundnut"])
    
    st.markdown("---")
    st.markdown("### 🎙️ Tamil Voice Assistant")
    st.markdown("Speak your question in Tamil:")
    if st.button("🎤 Start Listening"):
        st.info("Listening... (In a real app, this connects to the mic)")
        st.success("You asked: 'தக்காளி விலை எப்படி இருக்கும்?'")
        st.markdown("**Assistant:** தக்காளி விலை அடுத்த 7 நாட்களில் கிலோவுக்கு 4 ரூபாய் அதிகரிக்க வாய்ப்புள்ளது.")
        
    st.markdown("---")
    st.markdown("Or type your question (English or Tamil):")
    user_query = st.text_input("e.g. Onion price? / தக்காளி விலை?", label_visibility="collapsed")
    if user_query:
        st.success(f"You asked: '{user_query}'")
        q = user_query.lower()
        if "நோய்" in q or "disease" in q or "pest" in q or "blight" in q:
            st.markdown("**Assistant:** அதிக ஈரப்பதம் காரணமாக இலை கருகல் நோய் வர வாய்ப்புள்ளது. *(Due to high humidity, Leaf Blight is likely. Take precautions.)*")
        elif "onion" in q or "வெங்காயம்" in q:
            st.markdown("**Assistant:** வெங்காயம் விலை அடுத்த 7 நாட்களில் கிலோவுக்கு 2 ரூபாய் குறைய வாய்ப்புள்ளது. *(Onion prices are expected to drop by ₹2 per kg in the next 7 days. Sell early!)*")
        elif "groundnut" in q or "நிலக்கடலை" in q:
            st.markdown("**Assistant:** நிலக்கடலை விலை நிலையாக இருக்கும். *(Groundnut prices are expected to remain stable.)*")
        else:
            st.markdown("**Assistant:** தக்காளி விலை அடுத்த 7 நாட்களில் கிலோவுக்கு 4 ரூபாய் அதிகரிக்க வாய்ப்புள்ளது. *(Tomato prices are expected to rise by ₹4 per kg in the next 7 days.)*")

# Load specific crop model and data
xgb_model, rf_model = load_models(crop)
df_live = get_live_data(crop)

# --- Layout: 3 Pillars ---
tab1, tab2, tab3 = st.tabs(["📈 Live Price Forecast", "🦠 Live Disease Risk", "🌦️ Weather Insights"])

# --- TAB 1: Live Price Forecast ---
with tab1:
    st.subheader(f"Live 7-Day Price Forecast for {crop} in {village}")
    
    if df_live is not None and xgb_model is not None:
        last_row = df_live.iloc[-1]
        
        # Exact features expected by model
        features = [
            'Modal_Price', 'price_lag_1', 'price_lag_3', 'price_lag_7',
            'price_roll_mean_7', 'temperature_2m_max', 'temp_roll_mean_7', 
            'precipitation_sum', 'precip_roll_sum_7', 'relative_humidity_2m_mean',
            'day_of_year', 'month'
        ]
        
        X_live = last_row[features].to_frame().T.astype(float)
        predicted_price_quintal = xgb_model.predict(X_live)[0]
        
        current_price_kg = last_row['Modal_Price'] / 100.0
        pred_price_kg = predicted_price_quintal / 100.0
        diff = pred_price_kg - current_price_kg
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Live Price", f"₹ {current_price_kg:.2f} / kg", f"Updated: {last_row['date'].strftime('%d %b')}")
        col2.metric("Predicted Price (7 Days)", f"₹ {pred_price_kg:.2f} / kg", f"{'+' if diff > 0 else ''}₹{diff:.2f} (Forecast)")
        
        action = "HOLD" if diff > 1.5 else ("SELL" if diff < -1.5 else "STABLE")
        col3.metric("AI Action Recommendation", action, "Based on XGBoost output")
        
        # Plot last 14 days + Prediction
        hist_df = df_live.tail(14)[['date', 'Modal_Price']].copy()
        hist_df['Modal_Price'] = hist_df['Modal_Price'] / 100.0
        hist_df['Type'] = 'Historical'
        
        pred_date = last_row['date'] + pd.Timedelta(days=7)
        pred_row = pd.DataFrame({'date': [pred_date], 'Modal_Price': [pred_price_kg], 'Type': ['Forecast']})
        
        # Add a connecting line from today to forecast
        connector_row = pd.DataFrame({'date': [last_row['date']], 'Modal_Price': [current_price_kg], 'Type': ['Forecast']})
        
        plot_df = pd.concat([hist_df, connector_row, pred_row], ignore_index=True)
        
        fig = px.line(plot_df, x='date', y='Modal_Price', color='Type', 
                      color_discrete_map={'Historical': '#1f77b4', 'Forecast': '#ff7f0e'},
                      markers=True, title=f"Live {crop} Price Inference (₹ per kg)")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Live Data or Model missing. Please run data collection and training scripts first.")

# --- TAB 2: Live Disease Risk ---
with tab2:
    st.subheader("Live Biological Disease & Pest Alert")
    
    if df_live is not None and rf_model is not None:
        last_row = df_live.iloc[-1]
        
        # Features for Disease Model
        rf_features = last_row[['temperature_2m_max', 'relative_humidity_2m_mean', 'precipitation_sum']].to_frame().T.astype(float)
        predicted_disease = rf_model.predict(rf_features)[0]
        
        if "Healthy" in predicted_disease:
            st.markdown(f'<div class="alert-box alert-success"><strong>Current Status: {predicted_disease}</strong>. Weather conditions are safe.</div>', unsafe_allow_html=True)
        elif "Fungal" in predicted_disease:
            st.markdown(f'<div class="alert-box alert-danger"><strong>CRITICAL ALERT: High probability of {predicted_disease}</strong> in the next 5 days due to high humidity ({last_row["relative_humidity_2m_mean"]:.1f}%) and rainfall.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-box alert-warning"><strong>WARNING: {predicted_disease}</strong> detected due to dry and hot weather.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Live Weather Impact Drivers")
            st.write(f"- **Max Temp:** {last_row['temperature_2m_max']:.1f}°C")
            st.write(f"- **Humidity:** {last_row['relative_humidity_2m_mean']:.1f}%")
            st.write(f"- **Rainfall:** {last_row['precipitation_sum']:.1f} mm")
            
        with col2:
            st.markdown("### Suggested AI Actions")
            if "Fungal" in predicted_disease:
                st.checkbox("Apply protective fungicide spray tomorrow morning.")
                st.checkbox("Ensure proper field drainage; avoid over-irrigation.")
            elif "Pest" in predicted_disease:
                st.checkbox("Spray Neem oil or appropriate pesticide.")
                st.checkbox("Set up yellow sticky traps in the field.")
            else:
                st.checkbox("Maintain normal irrigation schedule.")
    else:
        st.error("Live Data or Model missing.")

# --- TAB 3: Weather Insights ---
with tab3:
    st.subheader("Agro-Meteorological Dashboard")
    
    weather_files = glob.glob("data/raw/weather_madurai_*.csv")
    if weather_files:
        weather_path = sorted(weather_files)[-1]
        df_weather_raw = pd.read_csv(weather_path)
        df_weather_raw['date'] = pd.to_datetime(df_weather_raw['time'])
        df_weather = df_weather_raw.tail(30) # Last 30 distinct days
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_weather['date'], y=df_weather['temperature_2m_max'], name='Max Temp (°C)', line=dict(color='firebrick'), mode='lines+markers'))
        fig3.add_trace(go.Bar(x=df_weather['date'], y=df_weather['precipitation_sum'], name='Rainfall (mm)', yaxis='y2', opacity=0.3))
        
        fig3.update_layout(
            title="Last 30 Days Weather (Live Data)",
            yaxis=dict(title='Temperature (°C)'),
            yaxis2=dict(title='Rainfall (mm)', overlaying='y', side='right', showgrid=False)
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.error("No weather data found in data/raw/. Please run the data collection scripts.")
