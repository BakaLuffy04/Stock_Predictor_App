import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load model and metrics
model = joblib.load("stock_lr_model.pkl")
metrics = joblib.load("metrics.pkl")  # contains 'rmse' and 'mae'

# Load recent data for plotting
df = pd.read_csv("stock_cleaned.csv")
features = ["Open", "High", "Low", "Volume", "Volatility", "DayOfWeek", "Day_Return"]
target = "Target_Close"

X = df[features]
y = df[target]
y_pred = model.predict(X)

# Streamlit UI
st.title("📈 Nifty50 Next-Day Stock Price Predictor")
st.write("Enter today's stock values to predict **tomorrow's closing price**.")

# User inputs
open_price = st.number_input("Open Price", min_value=0.0, value=20000.0)
high_price = st.number_input("High Price", min_value=0.0, value=20500.0)
low_price = st.number_input("Low Price", min_value=0.0, value=19800.0)
close_price = st.number_input("Close Price (today)", min_value=0.0, value=20100.0)
volume = st.number_input("Volume", min_value=0.0, value=1e7)

# Derived features
volatility = high_price - low_price
day_return = (close_price - open_price) / open_price
day_of_week = datetime.today().weekday()
input_vector = np.array([[open_price, high_price, low_price, volume, volatility, day_of_week, day_return]])

# Plot function
def plot_predictions(actual, predicted):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(actual, label='Actual', marker='o')
    ax.plot(predicted, label='Predicted', marker='x')
    ax.set_title("Recent 30 Predictions vs Actual")
    ax.set_xlabel("Day")
    ax.set_ylabel("Close Price")
    ax.legend()
    return fig

# Prediction
if st.button("🔮 Predict Tomorrow's Close"):
    prediction = model.predict(input_vector)[0]
    mae = metrics["mae"]
    st.success(f"📊 Predicted Closing Price: ₹{prediction:.2f}")
    st.info(f"✅ Confidence Interval: ₹{prediction - mae:.2f} – ₹{prediction + mae:.2f}")

    st.markdown("---")
    st.subheader("📉 Model Performance on Recent Data")
    st.write(f"**RMSE:** {metrics['rmse']:.2f} &nbsp;&nbsp;&nbsp; **MAE:** {mae:.2f}")

    st.pyplot(plot_predictions(y.tail(30).values, y_pred[-30:]))