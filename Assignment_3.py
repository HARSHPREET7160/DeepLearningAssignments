import pandas as pd
import numpy as np

df=pd.read_csv("glass.csv")

print(df.shape)
print(df.head())

df["y"]=(df["Type"]==1).astype(int)
df=df.drop(columns=["Type"])

X=df.drop(columns=["y"]).values
y=df["y"].values

np.random.seed(42)

indices=np.arange(len(X))
np.random.shuffle(indices)

split=int(0.8*len(X))

train_idx = indices[:split]
test_idx  = indices[split:]

X_train = X[train_idx]
X_test  = X[test_idx]

y_train = y[train_idx]
y_test  = y[test_idx]


mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test  = (X_test - mean) / std



def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def predict_proba(X, w, b):
    z = X @ w + b
    p = sigmoid(z)
    return p



def loss(y, p):
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))



def update_weights(X, y, w, b, lr):
    p = predict_proba(X, w, b)
    error = p - y

    w = w - lr * (X.T @ error) / len(y)
    b = b - lr * np.mean(error)

    return w, b


w = np.zeros(X_train.shape[1])
b = 0.0

lr = 0.1
epochs = 100

for _ in range(epochs):
    w, b = update_weights(X_train, y_train, w, b, lr)



def predict_label(p, threshold=0.5):
    return (p >= threshold).astype(int)




p_test = predict_proba(X_test, w, b)

y_pred_05 = predict_label(p_test, threshold=0.5)
y_pred_07 = predict_label(p_test, threshold=0.7)

acc_05 = np.mean(y_pred_05 == y_test)
acc_07 = np.mean(y_pred_07 == y_test)

print("Accuracy (threshold = 0.5):", acc_05)
print("Accuracy (threshold = 0.7):", acc_07)
