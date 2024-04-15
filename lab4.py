import random
import matplotlib.pyplot as plt


def create_2d_array(rows, columns):
    return [[0] * columns for _ in range(rows)]


def shuffle_array(array):
    shuffled_array = array[:]
    random.shuffle(shuffled_array)
    return shuffled_array


def train_hopfield_network(patterns):
    num_pixels = len(patterns[0])
    weights = create_2d_array(num_pixels, num_pixels)

    for pattern in patterns:
        pattern_copy = [-1 if p == 0 else p for p in pattern]

        for i in range(num_pixels):
            for j in range(num_pixels):
                if i != j:
                    weights[i][j] += pattern_copy[i] * pattern_copy[j]

    for i in range(num_pixels):
        weights[i][i] = 0

    # Normalize weights
    for i in range(num_pixels):
        for j in range(num_pixels):
            weights[i][j] /= num_pixels

    return weights


def recall_pattern(pattern, weights, num_iterations=1000):
    output = pattern[:]

    for _ in range(num_iterations):
        neurons_to_update = shuffle_array(list(range(len(pattern))))

        for neuron in neurons_to_update:
            net_input = sum(weights[neuron][idx] * state for idx, state in enumerate(output))
            output[neuron] = 1 if net_input > 0 else 0

    return output


def add_noise(pattern, noise_level):
    noisy_pattern = pattern[:]

    for index in range(len(noisy_pattern)):
        if random.random() < noise_level:
            noisy_pattern[index] *= 0

    return noisy_pattern


def print_pattern(pattern):
    pattern_str = ''
    for i in range(len(pattern)):
        if i % 7 == 0 and i != 0:
            pattern_str += '\n'
        pattern_str += str(pattern[i]) + ' '
    return pattern_str


def plot_pattern(pattern, title):
    if len(pattern) != 77:
        return "Invalid pattern size."

    image = [[1 - pattern[i * 7 + j] for j in range(7)] for i in range(11)]
    plt.figure(figsize=(2, 3.5))
    plt.imshow(image, cmap='gray', aspect='auto')
    plt.title(title)
    plt.axis('off')
    plt.grid(True)
    plt.show()

zero = [
    0, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 0,
]

one = [
    0, 0, 1, 1, 0, 0, 0,
    0, 1, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 0,
]

two = [
    0, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 0,
]


three = [
    1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 0,
]

four = [
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 1, 0, 0,
    0, 0, 1, 0, 1, 0, 0,
    0, 1, 0, 0, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0, 0,
]

five = [
    1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 0,
]

six = [
    0, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 0,
]

seven = [
    1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0,
]

eight = [
    0, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 0,
]

nine = [
    0, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 0,
]

patterns = [zero, one, two, three, four, five, six, seven, eight, nine]

for digit, pattern in enumerate(patterns):
    weights = train_hopfield_network([pattern])
    noisy_pattern = add_noise(pattern, noise_level=0.95)
    recovered_pattern = recall_pattern(noisy_pattern, weights)

    print(f"Noisy {digit}:")
    plot_pattern(noisy_pattern, f"Noisy {digit}")
    print(print_pattern(noisy_pattern))
    print(f"\nRecovered {digit}:")
    plot_pattern(recovered_pattern, f"Recovered {digit}")
    print(print_pattern(recovered_pattern))
    print()
