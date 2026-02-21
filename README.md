# Stock Price Prediction using LSTM

## Overview
This project implements stock price forecasting using:

- Deep LSTM model (Keras)
- Custom LSTM implemented from scratch (NumPy)
- Streamlit web application for visualization

## Structure

- notebooks/ → Experimental notebooks
- models/ → Saved models and custom LSTM implementation
- app/ → Streamlit deployment
- train_keras.py → Train Keras LSTM
- train_scratch.py → Train NumPy LSTM

## Run

Train Keras model:
python train_keras.py

Train NumPy LSTM:
python train_scratch.py

Run Web App:
streamlit run app/webapp.py