import numpy as np
from models.lstm_from_scratch import LSTM

# Dummy sine data (replace with real stock later if needed)
series = np.sin(np.linspace(0, 50, 1000))

window_size = 20
X = []
y = []

for i in range(window_size, len(series)):
    X.append(series[i-window_size:i])
    y.append(series[i])

X = np.array(X)
y = np.array(y)

model = LSTM(input_size=1, hidden_size=32, output_size=1)
model.train(X, y, epochs=5)