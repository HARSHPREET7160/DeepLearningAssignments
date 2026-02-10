import pandas as pd
import numpy as np

df=pd.read_csv("multiple_linear_regression_dataset.csv")

print(df.head())

X=df[["age", "experience"]].values
y=df["income"].values

n_features=X.shape[1]
w=np.zeros(n_features)
b=0.0

def predict(X, w, b):
    y_hat = X.dot(w) + b
    return y_hat


def mean_squared_error(y, y_hat):
    loss = ((y_hat - y) ** 2).mean()
    return loss


def compute_gradients(X, y, y_hat):
    N = len(y)

    dw = (2 / N) * X.T.dot(y_hat - y)
    db = (2 / N) * (y_hat - y).sum()

    return dw, db

def update_parameters(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return w, b


lr = 0.0001
epochs = 1000

for epoch in range(epochs):
    y_hat = predict(X, w, b)
    loss = mean_squared_error(y, y_hat)

    dw, db = compute_gradients(X, y, y_hat)
    w, b = update_parameters(w, b, dw, db, lr)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


print("Final weights:", w)
print("Final bias:", b)


new_candidate = np.array([35, 6])
predicted_salary = new_candidate.dot(w) + b

print("Predicted Salary:", predicted_salary)

