# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Load live BTC data from Yahoo Finance
def load_data():
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="10y", interval="1d")
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    df['Volatility_7d'] = df['Close'].rolling(window=7).std().fillna(method='bfill')
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean().fillna(method='bfill')
    df = df.dropna()
    return df

# Predict next 5 days

def predict_next_days(model, scaler_X, scaler_y, X, lookback, scale_y):
    future_days = 5
    future_preds = []
    last_seq = X[-lookback:].copy()
    current_seq = scaler_X.transform(last_seq).reshape(1, lookback, X.shape[1])

    for _ in range(future_days):
        pred_scaled = model.predict(current_seq)[0][0]
        future_preds.append(pred_scaled)

        new_row = last_seq.iloc[-1].copy()
        last_seq = pd.concat([last_seq, pd.DataFrame([new_row])])
        last_seq = last_seq.iloc[1:]
        last_seq = last_seq[scaler_X.feature_names_in_]  # align columns
        current_seq = scaler_X.transform(last_seq).reshape(1, lookback, X.shape[1])

    if scale_y:
        future_preds = scaler_y.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    last_date = X.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(future_days)]
    return future_dates, future_preds

# --- Streamlit App ---
st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")
st.title("📈 Bitcoin Price Prediction using LSTM")

model_option = st.selectbox("Choose an LSTM Model:", (
    "1-Day Lookback (y scaled)",
    "7-Day Lookback (y unscaled)",
    "5-Day Lookback (y unscaled)"
))

model_files = {
    "1-Day Lookback (y scaled)": ("models/lstm_1day_scaled.keras", "models/scaler_X_1day.pkl", "models/scaler_y_1day.pkl", 1, True),
    "7-Day Lookback (y unscaled)": ("models/lstm_7day_unscaled.keras", "models/scaler_X_7day.pkl", None, 7, False),
    "5-Day Lookback (y unscaled)": ("models/lstm_5day_unscaled.keras", "models/scaler_X_5day.pkl", None, 5, False),
}

with st.spinner("Loading model and generating predictions..."):
    df = load_data()
    model_path, scaler_X_path, scaler_y_path, lookback, scale_y = model_files[model_option]
    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path) if scale_y else None
    X = df.drop(columns=['Close'])
    future_dates, future_preds = predict_next_days(model, scaler_X, scaler_y, X, lookback, scale_y)

# Timeframe selector for actual price chart
timeframe_option = st.selectbox("Select timeframe for price chart:", ["All", "1Y", "6M", "3M", "1M"])
cutoffs = {
    "1Y": 365,
    "6M": 180,
    "3M": 90,
    "1M": 30
}

plot_df = df.copy()
if timeframe_option != "All":
    plot_df = plot_df.iloc[-cutoffs[timeframe_option]:]

# Plot actual + predicted
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df.index, plot_df['Close'], label='Actual Price', color='blue')
ax.plot(future_dates, future_preds, label='Predicted Price', linestyle='--', marker='o', color='orange')
ax.set_title("Bitcoin Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Arsh Gupta")
