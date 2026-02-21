import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import datetime
import os


STOCK = "BANKBARODA.NS"
START = "2000-01-01"
END = "2020-01-01"
WINDOW_SIZE = 100
EPOCHS = 10
MODEL_PATH = "models/keras_model.h5"


print("Downloading data...")
df = web.DataReader(STOCK, "yahoo", START, END)

data = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


X = []
y = []

for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i - WINDOW_SIZE:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

print("Training samples:", X.shape)


model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

model.summary()


print("Training model...")
model.fit(X, y, epochs=EPOCHS, batch_size=32)


if not os.path.exists("models"):
    os.makedirs("models")

model.save(MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")