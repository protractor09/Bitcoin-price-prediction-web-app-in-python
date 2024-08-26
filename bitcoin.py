import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

from sklearn.metrics import mean_absolute_error, mean_squared_error

ticker_symbol = 'BTC-USD'

btc_data = yf.Ticker(ticker_symbol)


st.write("## This is a webapp for seeing the current bitcoin prices and predicting it to a certain degree")

x=st.text_input("Historical data time period (e.g., 1d, 5d, 1mo, 3mo, 1y)")
hist = btc_data.history(period=x)

if x:
    try:
        st.write(hist)
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['Close'], label='Bitcoin Price')
        plt.title(f'Bitcoin (BTC-USD) Prices - {x}')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.legend()
        st.write(f"Graph for {x} amount of time")
        st.pyplot(plt)
    except Exception as e:
        st.write("Error encountered while trying to fetch the data")


current_price = btc_data.history(period='1d')['Close'].iloc[-1]

st.write("The current price of bitcoin is: ",current_price)

st.write("Potential bitcoin prices (accuracy around 50%): ")

btc_reset = hist.reset_index()
btc_reset['Date'] = btc_reset['Date'].dt.tz_localize(None)

btc_train = btc_reset.iloc[:len(btc_reset)-365]
btc_test = btc_reset.iloc[len(btc_reset)-365:]

btc_train = btc_train.rename(columns={'Date': 'ds', 'Close': 'y'})

model_p = Prophet()
model_p.fit(btc_train)

future = model_p.make_future_dataframe(periods=365)
forecast = model_p.predict(future)

fig1 = model_p.plot(forecast)
st.pyplot(fig1)

# Optionally, plot the forecast components
fig2 = model_p.plot_components(forecast)
st.pyplot(fig2)

# If you still want to see the DataFrame
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

