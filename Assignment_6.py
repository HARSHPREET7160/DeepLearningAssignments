import numpy as np
import pandas as pd

np.random.seed(42)

X = np.linspace(0, 10, 50).reshape(-1, 1)

noise = np.random.normal(0, 0.2, size=(50, 1))
y = np.log(X + 1) + noise

input_units = X.shape[1]

# Step 2 - Decide model shape
# Hidden units chosen: 3
# Why more than 1?
# One hidden unit is usually too limited to capture curve bending well.
# Why not too many?
# Too many hidden units can make training unstable and unnecessarily complex.
# Here we are fixing model shape, not tuning final parameter values.
hidden_units = 3
output_units = 1

W1 = np.random.uniform(-1, 1, size=(input_units, hidden_units))
b1 = np.zeros((1, hidden_units))
W2 = np.random.uniform(-1, 1, size=(hidden_units, output_units))
b2 = np.zeros((1, output_units))

def activation(z):
    return np.maximum(0, z)

def activation_slope(z):
    return (z > 0).astype(float)

learning_rate = 0.01
epochs = 1000

loss_history = []

for epoch in range(epochs):
    z1 = X @ W1 + b1
    h = activation(z1)
    y_hat = h @ W2 + b2

    error = y_hat - y

    loss = np.mean(error ** 2)
    loss_history.append(loss)

    dL_dy = 2 * error / len(X)

    dL_dW2 = h.T @ dL_dy
    dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)

    dL_dh = dL_dy @ W2.T

    dL_dz1 = dL_dh * activation_slope(z1)

    dL_dW1 = X.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

z1_final = X @ W1 + b1
h_final = activation(z1_final)
y_pred = h_final @ W2 + b2

print("\nFinal Loss:", float(loss_history[-1]))

result_df = pd.DataFrame(
    {
        "X": X.flatten(),
        "y_true": y.flatten(),
        "y_pred": y_pred.flatten(),
    }
)
result_df["abs_error"] = np.abs(result_df["y_true"] - result_df["y_pred"])

print("\nSample predictions:")
print(result_df.head(10).to_string(index=False))

history_df = pd.DataFrame({"epoch": np.arange(1, epochs + 1), "loss": loss_history})
print("\nLoss summary:")
print(history_df.describe().to_string())
