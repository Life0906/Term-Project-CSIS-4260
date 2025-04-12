import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set page configuration
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Bitcoin Price Prediction App")
st.markdown("""
This app uses LSTM models to predict Bitcoin prices based on historical data.
Choose a model and see the predictions for the next few days.
""")

# Load the data
@st.cache_data
def load_data():
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="10y", interval="1d")
    # Drop the unneeded columns
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    # Add technical indicators
    df = add_technical_indicators(df)
    return df

# Calculate technical indicators
def add_technical_indicators(data):
    # Function to calculate SMA
    def calculate_sma(data, period):
        return data['Close'].rolling(window=period).mean()
    
    # Function to calculate EMA
    def calculate_ema(data, period):
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    # Function to calculate RSI
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # Function to calculate MACD
    def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
        short_ema = calculate_ema(data, short_period)
        long_ema = calculate_ema(data, long_period)
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    # Calculate RSI
    data['RSI'] = calculate_rsi(data)
    
    # Calculate MACD
    macd, signal, hist = calculate_macd(data)
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    data['MACD_Hist'] = hist
    
    # Calculate EMAs
    data['EMA_8'] = calculate_ema(data, 8)
    data['EMA_21'] = calculate_ema(data, 21)
    data['EMA_55'] = calculate_ema(data, 55)
    data['EMA_200'] = calculate_ema(data, 200)
    
    # Daily Return
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Volatility
    data['Volatility_7d'] = data['Daily_Return'].rolling(window=7).std()
    
    # Relative Volume
    volume_rolling_avg_20d = data['Volume'].rolling(window=20).mean()
    data['Relative_Volume'] = data['Volume'] / volume_rolling_avg_20d
    
    # Drop NAs
    return data.dropna()

# Model class
class LSTMModel:
    def __init__(self, lookback_days):
        """Initialize LSTM model with specified lookback days."""
        self.lookback_days = lookback_days
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    def prepare_data(self, df):
        """Prepare data for training and prediction."""
        # Separate features and target
        X = df.drop(columns=['Close'])
        y = df['Close']
        
        # Scale features and target
        self.scaler_X.fit(X)
        self.scaler_y.fit(y.values.reshape(-1, 1))
        
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1))
        
        return X, y, X_scaled, y_scaled
    
    def create_sequences(self, X_scaled, y_scaled):
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.lookback_days):
            X_seq.append(X_scaled[i:i + self.lookback_days])
            y_seq.append(y_scaled[i + self.lookback_days])
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, 
                      input_shape=(input_shape[1], input_shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=25, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, df, epochs=100, batch_size=32):
        """Train the LSTM model."""
        # Prepare data
        X, y, X_scaled, y_scaled = self.prepare_data(df)
        
        # Create sequences
        if self.lookback_days > 1:
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
            X_train_lstm = X_seq.reshape(X_seq.shape[0], self.lookback_days, X_seq.shape[2])
            y_train_lstm = y_seq
        else:
            # For 1-day lookback
            X_train_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            y_train_lstm = y_scaled
        
        # Build model
        self.model = self.build_model(X_train_lstm.shape)
        
        # Train model
        history = self.model.fit(
            X_train_lstm, 
            y_train_lstm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        return history
    
    def predict_future(self, df, days=5):
        """Predict Bitcoin prices for the next specified days."""
        # Prepare data
        X = df.drop(columns=['Close'])
        
        # Scale features
        X_scaled = self.scaler_X.transform(X)
        
        # Get the last sequence
        if self.lookback_days > 1:
            last_sequence = X_scaled[-self.lookback_days:]
            input_seq = last_sequence.reshape(1, self.lookback_days, X_scaled.shape[1])
        else:
            # For 1-day lookback
            last_sequence = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])
            input_seq = last_sequence
        
        # Predict the next 'days' days
        predictions = []
        dates = []
        last_date = df.index[-1]
        
        for i in range(days):
            # Predict the next day's price (scaled)
            next_day_scaled = self.model.predict(input_seq)
            
            # Inverse transform to get the actual price
            next_day_price = self.scaler_y.inverse_transform(next_day_scaled)[0][0]
            
            # Add to predictions
            predictions.append(next_day_price)
            
            # Calculate next date
            next_date = last_date + timedelta(days=i+1)
            dates.append(next_date)
            
            # For multi-day lookback models, update the input sequence
            if self.lookback_days > 1:
                # For demonstration purposes, we'll use the same features except for the closing price
                # In a real application, you might want to update other features as well
                next_row_features = X_scaled[-1].copy()  # Copy the last day's features
                
                # Slide the window forward, remove first day and add prediction as next day
                input_seq = np.append(input_seq[:, 1:, :], 
                                     np.array([next_row_features]).reshape(1, 1, X_scaled.shape[1]), 
                                     axis=1)
            
        return dates, predictions

    def save_model(self, filepath):
        """Save the trained model."""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = load_model(filepath)

# Initialize session state to store models
if 'models' not in st.session_state:
    st.session_state.models = {
        '1-day': None,
        '5-day': None,
        '7-day': None
    }

# Load data
with st.spinner('Loading Bitcoin data...'):
    df = load_data()

# Sidebar
st.sidebar.header("Model Configuration")

# Model selection
model_option = st.sidebar.selectbox(
    "Select prediction model:",
    ['1-day LSTM', '5-day LSTM', '7-day LSTM']
)

# Number of future days to predict
future_days = st.sidebar.slider("Days to predict:", 1, 14, 5)

# Training option
if st.sidebar.button("Train Models"):
    with st.spinner('Training models, please wait...'):
        progress_bar = st.sidebar.progress(0)
        
        # Train 1-day model
        st.session_state.models['1-day'] = LSTMModel(lookback_days=1)
        history_1day = st.session_state.models['1-day'].train(df, epochs=50)
        progress_bar.progress(33)
        
        # Train 5-day model
        st.session_state.models['5-day'] = LSTMModel(lookback_days=5)
        history_5day = st.session_state.models['5-day'].train(df, epochs=50)
        progress_bar.progress(66)
        
        # Train 7-day model
        st.session_state.models['7-day'] = LSTMModel(lookback_days=7)
        history_7day = st.session_state.models['7-day'].train(df, epochs=50)
        progress_bar.progress(100)
        
        st.sidebar.success("All models trained successfully!")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Historical price chart
    st.subheader("Bitcoin Historical Price")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[-365:], df['Close'][-365:])
    ax.set_title('Bitcoin Price (Last 365 Days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True)
    st.pyplot(fig)

# Model prediction section
st.subheader("Bitcoin Price Predictions")

# Check if models are trained
if not any(model for model in st.session_state.models.values()):
    st.warning("Please train the models first by clicking the 'Train Models' button in the sidebar.")
else:
    # Get the selected model
    if model_option == '1-day LSTM':
        model = st.session_state.models['1-day']
        lookback = 1
    elif model_option == '5-day LSTM':
        model = st.session_state.models['5-day']
        lookback = 5
    else:  # 7-day LSTM
        model = st.session_state.models['7-day']
        lookback = 7
    
    if model:
        # Make predictions
        future_dates, future_prices = model.predict_future(df, days=future_days)
        
        # Display predictions
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price (USD)': future_prices
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(pred_df.style.format({'Predicted Price (USD)': '${:.2f}'}), height=400)
        
        with col2:
            # Prediction chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data (last 30 days)
            historical_days = 30
            ax.plot(df.index[-historical_days:], df['Close'][-historical_days:], 
                   label='Historical Price', color='blue')
            
            # Plot predictions
            ax.plot(future_dates, future_prices, label=f'Predicted Price ({lookback}-day model)', 
                  color='red', linestyle='--', marker='o')
            
            # Add a vertical line to separate historical from predicted
            ax.axvline(x=df.index[-1], color='green', linestyle='-', alpha=0.5, 
                     label='Prediction Start')
            
            ax.set_title(f'Bitcoin Price Prediction ({lookback}-day model)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)

# Model comparison section
st.subheader("Model Comparison")

if all(model for model in st.session_state.models.values()):
    # Get predictions from all models
    dates_1day, prices_1day = st.session_state.models['1-day'].predict_future(df, days=future_days)
    dates_5day, prices_5day = st.session_state.models['5-day'].predict_future(df, days=future_days)
    dates_7day, prices_7day = st.session_state.models['7-day'].predict_future(df, days=future_days)
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data (last 30 days)
    historical_days = 30
    ax.plot(df.index[-historical_days:], df['Close'][-historical_days:], 
           label='Historical Price', color='blue')
    
    # Plot predictions from each model
    ax.plot(dates_1day, prices_1day, label='1-day LSTM', color='red', 
          linestyle='--', marker='o')
    ax.plot(dates_5day, prices_5day, label='5-day LSTM', color='green', 
          linestyle='--', marker='x')
    ax.plot(dates_7day, prices_7day, label='7-day LSTM', color='purple', 
          linestyle='--', marker='s')
    
    # Add a vertical line to separate historical from predicted
    ax.axvline(x=df.index[-1], color='black', linestyle='-', alpha=0.5, 
             label='Prediction Start')
    
    ax.set_title('Bitcoin Price Prediction - Model Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True)
    ax.legend()
    
    st.pyplot(fig)
    
    # Create comparison table
    comparison_data = []
    for i in range(len(dates_1day)):
        comparison_data.append({
            'Date': dates_1day[i].strftime('%Y-%m-%d'),
            '1-day LSTM': f'${prices_1day[i]:.2f}',
            '5-day LSTM': f'${prices_5day[i]:.2f}',
            '7-day LSTM': f'${prices_7day[i]:.2f}'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, height=400)

# About section
with st.expander("About this app"):
    st.markdown("""
    ### How it works
    
    This app uses Long Short-Term Memory (LSTM) neural networks to predict Bitcoin prices based on historical data.
    
    Three different LSTM models are used:
    - **1-day model**: Uses the previous day's data to predict the next day's price
    - **5-day model**: Uses the previous 5 days of data to predict the next day's price
    - **7-day model**: Uses the previous 7 days of data to predict the next day's price
    
    ### Features used for prediction
    - Opening, High, Low, Close prices
    - Volume
    - Technical indicators (RSI, MACD, EMAs)
    - Volatility metrics
    - Relative volume
    
    ### Disclaimer
    This app is for educational purposes only. Cryptocurrency prices are highly volatile and predictions should not be used as financial advice.
    """)

# GitHub link
st.sidebar.markdown("---")
st.sidebar.info(
    "This app is part of a Bitcoin price prediction project. "
    "View the code on GitHub (add your repo link here)."
)
