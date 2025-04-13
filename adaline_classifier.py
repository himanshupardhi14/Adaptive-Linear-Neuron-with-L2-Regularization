



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=100, regularization_strength=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = regularization_strength

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)  # Initialize weights and bias
        self.loss_history = []

        for _ in range(self.epochs):
            net_input = self.net_input(X)
            errors = y - net_input
            self.weights[1:] += self.learning_rate * (X.T.dot(errors) + self.regularization_strength * self.weights[1:])
            self.weights[0] += self.learning_rate * errors.sum()
            loss = (errors**2).mean() + self.regularization_strength * np.sum(self.weights[1:]**2)
            self.loss_history.append(loss)

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Adaline model with regularization
adaline = Adaline(learning_rate=0.01, epochs=100, regularization_strength=0.01)
adaline.fit(X_train, y_train)

# Make predictions
y_pred = adaline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(range(1, adaline.epochs + 1), adaline.loss_history, marker='o', label='Loss')
plt.title('Adaline Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Loss)')
plt.legend()
plt.grid()
plt.show()

# Plot accuracy as a bar chart
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy])
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()