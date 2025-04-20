import numpy as np
import matplotlib.pyplot as plt

# Generate some predictions and true values
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.8, 0.7, 0.3, 0.9])

# Mean Squared Error (MSE) Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Calculate losses
mse = mse_loss(y_true, y_pred)
bce = binary_cross_entropy(y_true, y_pred)

print("Mean Squared Error:", mse)
print("Binary Cross-Entropy:", bce)

# Visualize different loss functions
x = np.linspace(0, 1, 100)
y_true_plot = np.ones_like(x)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, (y_true_plot - x) ** 2)
plt.title('MSE Loss')
plt.xlabel('Prediction')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(x, -(y_true_plot * np.log(x + 1e-15)))
plt.title('Cross-Entropy Loss')
plt.xlabel('Prediction')
plt.ylabel('Loss')

plt.tight_layout()
plt.show() 