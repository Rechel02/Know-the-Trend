import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.hidden_size = hidden_size
        self.lr = lr

        def init_weight(rows, cols):
            return np.random.randn(rows, cols) * np.sqrt(1 / cols)

        self.Wf = init_weight(hidden_size, hidden_size + input_size)
        self.Wi = init_weight(hidden_size, hidden_size + input_size)
        self.Wc = init_weight(hidden_size, hidden_size + input_size)
        self.Wo = init_weight(hidden_size, hidden_size + input_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        self.Wy = init_weight(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, X):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        self.cache = []
        for x in X:
            x = x.reshape(-1, 1)
            combined = np.vstack((h, x))

            f = sigmoid(self.Wf @ combined + self.bf)
            i = sigmoid(self.Wi @ combined + self.bi)
            c_tilde = np.tanh(self.Wc @ combined + self.bc)
            c = f * c + i * c_tilde
            o = sigmoid(self.Wo @ combined + self.bo)
            h = o * np.tanh(c)

            self.cache.append((combined, f, i, c_tilde, c, o, h))

        y = self.Wy @ h + self.by
        return y

    def train(self, X_train, y_train, epochs=5):
        for epoch in range(epochs):
            total_loss = 0
            for X, y in zip(X_train, y_train):
                y = np.array([[y]])
                y_pred = self.forward(X)
                loss = mse(y, y_pred)
                total_loss += loss
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train):.6f}")