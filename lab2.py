import numpy as np

class KohonenNetwork:
    def __init__(self, shape):
        self.weights = np.random.rand(*shape)

    @staticmethod
    def normalize_data(data):
        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)
        scaling = -min_values / (max_values - min_values + 1e-10)
        return data / (max_values - min_values + 1e-10) + scaling

    def train(self, X, epochs, alpha, learning_rate):
        X = self.normalize_data(X)
        D = X.shape[1]
        N = self.weights.shape[0]
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                winner = np.argmin(np.linalg.norm(X[i] - self.weights, axis=1))
                for j in range(N):
                    distance = np.linalg.norm(j - winner)
                    self.weights[j] += alpha * np.exp(-distance / (2 * N * N)) * (X[i] - self.weights[j])
            alpha *= learning_rate
        return self.weights

    @staticmethod
    def classify_samples(samples, class1_avg, class2_avg):
        labels = []
        for sample in samples:
            dist1 = np.sum((sample - class1_avg) ** 2)
            dist2 = np.sum((sample - class2_avg) ** 2)
            label = 1 if dist1 < dist2 else 2
            labels.append(label)
        return labels

# Example usage
kohonen_network = KohonenNetwork(shape=(2, 5))
training_data = np.array([
    [5.4, 4020, 2060, 57, 37],
    [8.9, 4810, 2223, 140, 40],
    [19.5, 5380, 2910, 285, 60],
    [25, 5890, 2880, 300, 40],
    [44.8, 6870, 3270, 700, 55],
    [56, 6316, 3705, 700, 44],
    [68, 7380, 3755, 700, 38],
    [13.8, 5200, 2470, 300, 60],
    [9.2, 4285, 2348, 140, 42],
    [30, 5920, 3000, 500, 54],
    [31.8, 6070, 3180, 500, 60],
    [47.5, 6675, 3320, 600, 34],
    [44.2, 6770, 3070, 520, 37]
])
epochs = 100
alpha = 0.1
learning_rate = 0.9

final_weights = kohonen_network.train(training_data, epochs, alpha, learning_rate)

samples_to_classify = np.array([[46, 6770, 3070, 520, 37], [49, 6900, 3150, 520, 40]])
class1_avg = np.array([11.5, 4747, 2410, 188, 48])
class2_avg = np.array([44, 6578, 3231, 554, 43])

classified_labels = kohonen_network.classify_samples(samples_to_classify, class1_avg, class2_avg)

# Output the classified labels for objects 14 and 15
for i, sample in enumerate(samples_to_classify, start=14):
    class_label = "Class 1" if classified_labels[i-14] == 1 else "Class 2"
    print(f"Sample {i}: predicted class is {class_label}")
