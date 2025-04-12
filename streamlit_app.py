# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# --- Feature Engineering Utilities ---
def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

# Load live BTC data from Yahoo Finance

def load_data():
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="10y", interval="1d")
    df = df.drop(columns=['Dividends', 'Stock Splits'])

    df['RSI'] = calculate_rsi(df)
    macd_line = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['SMA_10'] = calculate_sma(df, 10)
    df['SMA_20'] = calculate_sma(df, 20)
    df['SMA_50'] = calculate_sma(df, 50)
    df['SMA_200'] = calculate_sma(df, 200)
    df['EMA_8'] = calculate_ema(df, 8)
    df['EMA_21'] = calculate_ema(df, 21)
    df['EMA_55'] = calculate_ema(df, 55)
    df['EMA_200'] = calculate_ema(df, 200)

    df['SMA_10_20_cross'] = (df['SMA_10'] > df['SMA_20']).astype(int)
    df['SMA_10_20_cross_lagged'] = df['SMA_10_20_cross'].shift(1).fillna(0).astype(int)
    df['SMA_10_20_crossover_signal'] = df['SMA_10_20_cross'] - df['SMA_10_20_cross_lagged']

    df['SMA_20_50_cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)
    df['SMA_20_50_cross_lagged'] = df['SMA_20_50_cross'].shift(1).fillna(0).astype(int)
    df['SMA_20_50_crossover_signal'] = df['SMA_20_50_cross'] - df['SMA_20_50_cross_lagged']

    df['SMA_50_200_cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    df['SMA_50_200_cross_lagged'] = df['SMA_50_200_cross'].shift(1).fillna(0).astype(int)
    df['SMA_50_200_crossover_signal'] = df['SMA_50_200_cross'] - df['SMA_50_200_cross_lagged']

    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    df['Volatility_7d'] = df['Close'].rolling(window=7).std().fillna(method='bfill')
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean().fillna(method='bfill')

    df = df.dropna()
    return df

required_features = [
    'Open', 'High', 'Low', 'Volume',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'EMA_8', 'EMA_21', 'EMA_55', 'EMA_200',
    'SMA_10_20_crossover_signal',
    'SMA_20_50_crossover_signal',
    'SMA_50_200_crossover_signal',
    'Daily_Return', 'Volatility_7d', 'Relative_Volume'
]

# Predict next 5 days

def predict_next_days(model, scaler_X, scaler_y, X, lookback, scale_y):
    future_days = 5
    future_preds = []

    if lookback == 1:
        last_row = X.iloc[-1:][required_features]
        current_input = scaler_X.transform(last_row).reshape(1, 1, -1)
        for _ in range(future_days):
            pred_scaled = model.predict(current_input)[0][0]
            pred = pred_scaled
            future_preds.append(pred)
    else:
        last_seq_df = X[-lookback:].copy()
        current_seq = scaler_X.transform(last_seq_df[required_features]).reshape(1, lookback, -1)
        for _ in range(future_days):
            next_pred = model.predict(current_seq)[0][0]
            
            future_preds.append(next_pred)

            # Slide window with static dummy row (like your training script)
            new_input = last_seq_df.iloc[-1].copy()
            last_seq_df = pd.concat([last_seq_df, pd.DataFrame([new_input], columns=last_seq_df.columns)])
            last_seq_df = last_seq_df.iloc[1:]
            current_seq = scaler_X.transform(last_seq_df[required_features]).reshape(1, lookback, -1)

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
    "7-Day Lookback (y unscaled)": ("models/lstm_7day_unscaled.keras", "models/scaler_X_1day.pkl", None, 7, False),
    "5-Day Lookback (y unscaled)": ("models/lstm_5day_unscaled.keras", "models/scaler_X_1day.pkl", None, 5, False),
}

with st.spinner("Loading model and generating predictions..."):
    df = load_data()
    model_path, scaler_X_path, scaler_y_path, lookback, scale_y = model_files[model_option]
    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path) if scaler_y_path else None
    X = df[required_features]
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

# --- Display Forecasted Prices ---
st.subheader("📅 5-Day Price Forecast")
forecast_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
    'Predicted Price (USD)': [f"${p:,.2f}" for p in future_preds]
})
st.dataframe(forecast_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Paragon Peak")
