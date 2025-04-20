import numpy as np
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create and train models
l1_model = Lasso(alpha=1.0)  # L1 regularization
l2_model = Ridge(alpha=1.0)  # L2 regularization

l1_model.fit(X, y)
l2_model.fit(X, y)

# Make predictions
X_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_l1_pred = l1_model.predict(X_test)
y_l2_pred = l2_model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_l1_pred, color='red', label='L1 (Lasso)')
plt.plot(X_test, y_l2_pred, color='green', label='L2 (Ridge)')
plt.title('L1 vs L2 Regularization')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print("L1 (Lasso) coefficients:", l1_model.coef_)
print("L2 (Ridge) coefficients:", l2_model.coef_) 