import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Create dataset
def create_dataset(use_dataset=False, dimensions=2, sample_count=300):
    if use_dataset:
        X, y = make_blobs(n_samples=sample_count, centers=2, n_features=dimensions, random_state=42)
    else:
        X = np.random.rand(sample_count, dimensions) * 10 - 5
        y = np.array([0 if x[0] < 0 else 1 for x in X])
    return X, y

# Perceptron class
class Perceptron:
    def __init__(self, shape, use_sign=False):
        self.weights = np.random.rand(shape)
        self.bias = np.random.rand(1)
        self.shape = shape
        self.use_sign = use_sign

    def __repr__(self):
        return f"Perceptron({self.shape})"

    @staticmethod
    def relu_activation(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid_activation(x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y, epochs=10, learning_rate=0.01):
        for _ in range(epochs):
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])
                self.weights += learning_rate * (y[i] - y_pred) * X[i]
                self.bias += learning_rate * (y[i] - y_pred)
        return self

    def predict(self, x):
        if self.use_sign:
            return self.sigmoid_activation(np.dot(x, self.weights) + self.bias[0]) > 0.5
        return self.relu_activation(np.dot(x, self.weights) + self.bias[0]) > 0

# Visualize data
def visualize_data(X, y, points=None, is_3d=False):
    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], label="Class 0", alpha=0.5)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], label="Class 1", alpha=0.5)
        if points is not None:
            ax.plot_surface(points[0], points[1], points[2], alpha=0.5, label="Decision boundary")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        plt.legend()
        plt.show()
    else:
        plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0", alpha=0.5)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1", alpha=0.5)
        if points is not None:
            plt.plot(points[0], points[1], color="black", label="Decision boundary")
        plt.xlim(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1)
        plt.ylim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

def run_experiment(use_dataset, dimensions, sample_count, epochs, learning_rate, experiment_number, description):
    X, y = create_dataset(use_dataset=use_dataset, dimensions=dimensions, sample_count=sample_count)
    perceptron = Perceptron(shape=dimensions)
    perceptron.train(X, y, epochs=epochs, learning_rate=learning_rate)
    print(f"\nExperiment {experiment_number}:")
    print(f"Description: {description}")
    print(f"Number of epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weights: {perceptron.weights}")
    print(f"Bias: {perceptron.bias}")
    accuracy = np.mean(perceptron.predict(X) == y)
    print(f"Accuracy: {accuracy}")
    visualize_data(X, y, is_3d=(dimensions==3))

# Example usage
run_experiment(use_dataset=True, dimensions=2, sample_count=300, epochs=10, learning_rate=0.01, experiment_number=1, description="Case 2D with sklearn dataset")
run_experiment(use_dataset=False, dimensions=2, sample_count=300, epochs=10, learning_rate=0.01, experiment_number=2, description="Case 2D with random dataset in square")
run_experiment(use_dataset=False, dimensions=2, sample_count=300, epochs=30, learning_rate=0.01, experiment_number=3, description="Similar case to 2), but number of epochs 30")
run_experiment(use_dataset=True, dimensions=3, sample_count=300, epochs=10, learning_rate=0.01, experiment_number=4, description="Case 3D with sklearn dataset")
run_experiment(use_dataset=False, dimensions=3, sample_count=300, epochs=10, learning_rate=0.01, experiment_number=5, description="Case 3D with random dataset in cube")
