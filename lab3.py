import numpy as np
import sys
import math
import time
import matplotlib.pyplot as plt

class SOM:
    def __init__(self):
        self.trained = False

    def initialize_map(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels

    def save_map(self, filename):
        if self.trained:
            np.save(filename, self.node_vectors)
            return True
        return False

    def load_map(self, filename):
        self.node_vectors = np.load(filename)
        self.width, self.height, self.channels = self.node_vectors.shape
        self.trained = True
        return True

    def get_map_vectors(self):
        return self.node_vectors if self.trained else False

    def calculate_distance(self, vect_a, vect_b):
        if self.dist_method == 'euclidean':
            return np.linalg.norm(vect_a - vect_b)
        elif self.dist_method == 'cosine':
            return 1. - np.dot(vect_a, vect_b) / (np.linalg.norm(vect_a) * np.linalg.norm(vect_b))
        return None

    def find_matching_nodes(self, input_array):
        if not self.trained:
            return False

        n_data = input_array.shape[0]
        locations = np.zeros((n_data, 2), dtype=np.int32)
        distances = np.zeros(n_data, dtype=np.float32)

        for idx in range(n_data):
            data_vector = input_array[idx]
            min_dist, x, y = None, None, None
            for y_idx in range(self.height):
                for x_idx in range(self.width):
                    node_vector = self.node_vectors[y_idx, x_idx]
                    dist = self.calculate_distance(data_vector, node_vector)
                    if min_dist is None or min_dist > dist:
                        min_dist, x, y = dist, x_idx, y_idx

            locations[idx, 0], locations[idx, 1] = y, x
            distances[idx] = min_dist

        print('Done')
        return locations

    def initialize_weights(self):
        ds_mul = np.mean(self.input_array) / 0.5
        self.node_vectors = np.random.rand(self.height, self.width, self.channels) * ds_mul

    def train(self, input_array, n_iterations, batch_size=32, learning_rate=0.25, random_sampling=1.0,
              neighbor_distance=None, distance_method='euclidean'):
        self.input_array = input_array
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.dist_method = distance_method

        start_time = time.time()
        self.initialize_weights()

        self.learning_rate = learning_rate
        self.lr_decay = 0.8
        if neighbor_distance is None:
            neighbor_distance = min(self.width, self.height) / 1.3
        self.neighbor_distance = int(neighbor_distance)
        self.nb_decay = 1.5

        tmp_node_vectors = np.zeros((self.height + 2 * self.neighbor_distance, self.width + 2 * self.neighbor_distance, self.channels))
        tmp_node_vectors[self.neighbor_distance: self.neighbor_distance + self.height,
        self.neighbor_distance: self.neighbor_distance + self.width] = self.node_vectors.copy()
        self.node_vectors = tmp_node_vectors

        if random_sampling > 1 or random_sampling <= 0:
            random_sampling = 1
        n_data_points = int(self.input_array.shape[0] * random_sampling)

        data_indices = np.arange(self.input_array.shape[0])
        batch_count = math.ceil(n_data_points / self.batch_size)

        for iteration in range(self.n_iterations):
            self.update_neighbor_function(iteration)
            np.random.shuffle(data_indices)

            total_distance = 0
            total_count = 0

            for batch in range(batch_count):
                steps_left = n_data_points - batch * self.batch_size
                steps_in_batch = steps_left if steps_left < self.batch_size else self.batch_size

                bm_node_indices = np.zeros((steps_in_batch, 3), dtype=np.int32)

                for step in range(steps_in_batch):
                    total_count += 1

                    input_index = data_indices[batch * self.batch_size + step]
                    input_vector = self.input_array[input_index]
                    y, x, dist = self.find_best_matching_node(input_vector)
                    bm_node_indices[step, 0], bm_node_indices[step, 1], bm_node_indices[step, 2] = y, x, input_index
                    total_distance += dist

                self.update_node_vectors(bm_node_indices)

            print(f' Average distance = {total_distance / n_data_points:0.5f}')
            self.learning_rate *= self.lr_decay

        self.node_vectors = self.node_vectors[self.neighbor_distance: self.neighbor_distance + self.height,
                            self.neighbor_distance: self.neighbor_distance + self.width]

        del self.input_array

        end_time = time.time()
        self.trained = True
        print(f'Training done in {end_time - start_time:0.6f} seconds.')

    def update_node_vectors(self, bm_node_indices):
        for idx in range(bm_node_indices.shape[0]):
            node_y, node_x, input_index = bm_node_indices[idx, 0], bm_node_indices[idx, 1], bm_node_indices[idx, 2]
            input_vector = self.input_array[input_index]

            old_coeffs = self.node_vectors[node_y + self.y_delta + self.neighbor_distance, node_x + self.x_delta + self.neighbor_distance]

            update_vector = self.nb_weights * self.learning_rate * (np.expand_dims(input_vector, axis=0) - old_coeffs)

            self.node_vectors[node_y + self.y_delta + self.neighbor_distance,
            node_x + self.x_delta + self.neighbor_distance, :] += update_vector

    def find_best_matching_node(self, data_vector):
        min_dist, x, y = None, None, None
        for y_idx in range(self.height):
            for x_idx in range(self.width):
                node_vector = self.node_vectors[y_idx + self.neighbor_distance, x_idx + self.neighbor_distance]
                dist = self.calculate_distance(data_vector, node_vector)
                if min_dist is None or min_dist > dist:
                    min_dist, x, y = dist, x_idx, y_idx

        return y, x, min_dist

    def update_neighbor_function(self, iteration):
        size = self.neighbor_distance * 2
        sigma = size / (7 + iteration / self.nb_decay)
        self.nb_weights = np.full((size * size, self.channels), 0.0)
        cp = size / 2.0
        p1 = 1.0 / (2 * math.pi * sigma ** 2)
        pdiv = 2.0 * sigma ** 2
        y_delta, x_delta = [], []
        for y in range(size):
            for x in range(size):
                ep = -1.0 * ((x - cp) ** 2.0 + (y - cp) ** 2.0) / pdiv
                value = p1 * math.e ** ep
                self.nb_weights[y * size + x] = value
                y_delta.append(y - int(cp))
                x_delta.append(x - int(cp))
        self.x_delta = np.array(x_delta, dtype=np.int32)
        self.y_delta = np.array(y_delta, dtype=np.int32)

        self.nb_weights -= self.nb_weights[size // 2]
        self.nb_weights[self.nb_weights < 0] = 0
        self.nb_weights /= np.max(self.nb_weights)

    def get_umatrix(self):
        if not self.trained:
            return False

        umatrix = np.zeros((self.height, self.width))

        for map_y in range(self.height):
            for map_x in range(self.width):
                n_dist = 0
                total_dist = 0

                if map_y > 0:
                    dist_up = self.calculate_distance(self.node_vectors[map_y, map_x],
                                                      self.node_vectors[map_y - 1, map_x])
                    total_dist += dist_up
                    n_dist += 1

                if map_y < self.height - 1:
                    dist_down = self.calculate_distance(self.node_vectors[map_y, map_x],
                                                        self.node_vectors[map_y + 1, map_x])
                    total_dist += dist_down
                    n_dist += 1

                if map_x > 0:
                    dist_left = self.calculate_distance(self.node_vectors[map_y, map_x],
                                                        self.node_vectors[map_y, map_x - 1])
                    total_dist += dist_left
                    n_dist += 1

                if map_x < self.width - 1:
                    dist_right = self.calculate_distance(self.node_vectors[map_y, map_x],
                                                         self.node_vectors[map_y, map_x + 1])
                    total_dist += dist_right
                    n_dist += 1

                avg_dist = total_dist / n_dist
                umatrix[map_y, map_x] = avg_dist

        return umatrix

    def get_component_plane(self, component):
        return self.node_vectors[:, :, component].copy() if self.trained else False


def plot_data_on_map(data_locations, data_colors, width, height, node_width=20, data_marker_size=100):
    map_x, map_y = width, height

    canvas = np.ones((map_y * node_width, map_x * node_width))
    canvas[:node_width, :] = 0
    canvas[:, :node_width] = 0
    canvas[0, :] = 0
    canvas[-1, :] = 0
    canvas[:, 0] = 0
    canvas[:, -1] = 0

    plt.figure(figsize=(map_x, map_y))
    plt.imshow(canvas, cmap='Blues', interpolation='hanning')
    plt.axis('off')
    item_count_map = np.zeros((width, height))
    n_data_points = data_locations.shape[0]

    for i in range(n_data_points):
        x, y = data_locations[i, 1], data_locations[i, 0]
        items_in_cell = item_count_map[y, x]
        item_count_map[y, x] += 1
        x = x * node_width + node_width // 2 + items_in_cell * 5
        y = y * node_width + node_width // 2 + items_in_cell * 5
        plt.scatter(x, y, s=data_marker_size, edgecolors=[1, 0.2, 0.4])
        plt.text(x, y, str(i + 1), color='black', ha='center', va='center')
        plt.axis('off')

    plt.show()


data = [
    [6.5, 5020, 3060, 67, 47],
    [9.9, 5810, 3223, 240, 50],
    [20.5, 6380, 3910, 385, 70],
    [26, 6890, 3880, 400, 50],
    [45.8, 7870, 4270, 800, 65],
    [57, 7316, 4705, 800, 54],
    [69, 8380, 4755, 800, 48],
    [14.8, 6200, 3470, 400, 70],
    [10.2, 5285, 3348, 240, 52],
    [31, 6920, 4000, 600, 64],
    [32.8, 7070, 4180, 600, 70],
    [48.5, 7675, 4320, 700, 44],
    [45.2, 7770, 4070, 620, 47],
    [47, 7770, 4070, 620, 47],
]



dataset = np.array(data, dtype=np.float32)

colors = dataset.copy()
colors -= np.min(colors)
colors /= np.max(colors)

som = SOM()
som.initialize_map(width=20, height=20, channels=5)

som.train(dataset, random_sampling=0.5, n_iterations=6)

data_locations = som.find_matching_nodes(dataset)

plot_data_on_map(data_locations, width=20, height=20, data_colors=colors,
                 node_width=50, data_marker_size=500)