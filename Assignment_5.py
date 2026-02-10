import pandas as pd
import numpy as np

df=pd.read_csv("abalone.data", header=None)
print(df.head(5))

df.columns=[
    "Sex", "Length", "Diameter", "Height",
    "WholeWeight", "ShuckedWeight",
    "VisceraWeight", "ShellWeight", "Rings"
]

print("Number of rows:", len(df))
print("Column names:", df.columns)
print(df.head())


y=df["Rings"].values+1.5

X = df[["Length", "Diameter", "ShellWeight"]].values

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

split = int(0.8 * len(X))

train_idx = indices[:split]
test_idx  = indices[split:]

X_train = X[train_idx]
X_test  = X[test_idx]

y_train = y[train_idx]
y_test  = y[test_idx]

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape :", X_test.shape, y_test.shape)

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test  = (X_test - mean) / std


def forward(X, w, b):
    y_hat = X.dot(w) + b
    return y_hat

print("X shape:", X_train.shape)
print("w shape:", (3,))
print("b shape:", ())

def mse(y, y_hat):
    loss = np.mean((y_hat - y) ** 2)
    return loss


def grad_w(X, y, y_hat):
    N = len(y)
    dW = (2 / N) * X.T.dot(y_hat - y)
    return dW

def grad_b(y, y_hat):
    N = len(y)
    db = (2 / N) * np.sum(y_hat - y)
    return db


w = np.random.randn(3) * 0.01
b = 0.0

lr = 0.01
epochs = 1000

for epoch in range(epochs):
    y_hat = forward(X_train, w, b)
    loss = mse(y_train, y_hat)

    dW = grad_w(X_train, y_train, y_hat)
    db = grad_b(y_train, y_hat)

    w = w - lr * dW
    b = b - lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")



y_test_hat = forward(X_test, w, b)

test_mse = mse(y_test, y_test_hat)
test_mae = np.mean(np.abs(y_test - y_test_hat))

print("Test MSE:", test_mse)
print("Test MAE:", test_mae)


for i in range(5):
    print(
        "True age:", y_test[i],
        "Predicted age:", y_test_hat[i],
        "Absolute error:", abs(y_test[i] - y_test_hat[i])
    )
