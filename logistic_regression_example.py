import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate a random binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Plot decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Make predictions
print("Accuracy:", model.score(X, y)) 