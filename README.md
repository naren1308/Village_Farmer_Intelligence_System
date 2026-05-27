# 🌾 Village Farmer Intelligence System (VFIS)

**An AI-driven agronomy platform built for small/marginal farmers in Tamil Nadu, India.**

This project moves beyond toy datasets to solve real-world agricultural problems using **historical market data, biological weather correlations, and Generative AI voice interfaces.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Time--Series-green)
![Scikit-Learn](https://img.shields.io/badge/Random_Forest-Classification-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

---

## 🚀 The Three Pillars of Intelligence

Most agriculture apps stop at "crop recommendation." VFIS tackles the actual economic and biological challenges farmers face:

### 1. Market Price Forecasting (XGBoost)
* **Problem:** Farmers don't know whether to sell their highly perishable crops (like Tomatoes) today or wait a week.
* **Solution:** An XGBoost Regressor trained on **78,000+ daily price records** across Tamil Nadu Mandis, fused with historical weather data. 
* **Outcome:** Predicts the price of a crop 7 days into the future with an MAE of ₹3.40 per kg, allowing farmers to optimize their harvest timing for maximum profit.

### 2. Biological Disease Risk (Random Forest)
* **Problem:** Fungal diseases and pest attacks destroy crops rapidly based on sudden weather shifts.
* **Solution:** A Random Forest Classifier that analyzes the correlation between Temperature, Humidity, and Rainfall.
* **Outcome:** Triggers early-warning alerts for specific outbreaks (e.g., *Leaf Blight* during high humidity/temp spikes, or *Whitefly attacks* during dry/hot spells) with 99% accuracy.

### 3. Voice-Native Generative AI (LLM & RAG)
* **Problem:** Many village farmers cannot read complex dashboards or type queries on a smartphone.
* **Solution:** A Tamil-native voice assistant utilizing Speech-to-Text and Text-to-Speech APIs.
* **Outcome:** Farmers can press a button, ask a question in spoken Tamil (e.g., *"தக்காளி விலை எப்படி இருக்கும்?"*), and receive an AI-generated spoken response translating the complex ML predictions into simple advice.

---

## 🛠️ Architecture & Tech Stack

* **Data Engineering:** `pandas`, `glob`, `requests` (Open-Meteo API & Agmarknet historical data)
* **Machine Learning:** `xgboost` (Regression), `scikit-learn` (Random Forest Classification)
* **Voice AI Pipeline:** `SpeechRecognition`, `gTTS`, `playsound`
* **Frontend:** `streamlit`, `plotly`

## 📂 Project Structure
```text
Village_Farmer_Intelligence_System/
├── app/
│   └── dashboard.py               # Streamlit frontend
├── data/
│   ├── processed/                 # Cleaned CSVs ready for training
│   └── raw/                       # Raw Agmarknet and Weather CSVs
├── models/                        # Saved .pkl model files
├── src/
│   ├── data_collection/           # Scripts to fetch Weather and filter Kaggle data
│   ├── models/                    # Scripts to train XGBoost and Random Forest
│   └── rag/                       # Voice assistant logic (Speech-to-Text -> RAG -> TTS)
├── .env.example                   # API keys template
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies
```

## 💻 How to Run Locally

1. **Clone & Install**
   ```bash
   pip install -r requirements.txt
   ```
2. **Collect Data & Train Models**
   ```bash
   python src/data_collection/fetch_weather.py
   python src/data_collection/filter_dataset.py
   python src/models/train_price_model.py
   python src/models/train_disease_model.py
   ```
3. **Launch the Dashboard**
   ```bash
   streamlit run app/dashboard.py
   ```

---
*Built as a Data Science Portfolio Project focusing on real-world impact and end-to-end ML engineering.*
