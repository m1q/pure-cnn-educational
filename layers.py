"""
Pure Python CNN layers.
No NumPy. No PyTorch. Only standard Python.

Tensor convention for images:
    x[batch][channel][height][width]
"""

import math
import random


def zeros_4d(batch, channels, height, width):
    return [
        [
            [[0.0 for _ in range(width)] for _ in range(height)]
            for _ in range(channels)
        ]
        for _ in range(batch)
    ]


def shape4(x):
    return len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])


def pad4d(x, padding):
    if padding == 0:
        return x

    batch, channels, height, width = shape4(x)
    out = zeros_4d(batch, channels, height + 2 * padding, width + 2 * padding)

    for b in range(batch):
        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    out[b][c][i + padding][j + padding] = x[b][c][i][j]

    return out


def unpad4d(x, padding):
    if padding == 0:
        return x

    batch, channels, height, width = shape4(x)
    out = zeros_4d(batch, channels, height - 2 * padding, width - 2 * padding)

    for b in range(batch):
        for c in range(channels):
            for i in range(height - 2 * padding):
                for j in range(width - 2 * padding):
                    out[b][c][i][j] = x[b][c][i + padding][j + padding]

    return out


class Conv2D:
    """
    Convolution layer.

    Educational idea:
    - Each filter is a small matrix.
    - It slides over the image.
    - High output values mean: this filter found a useful pattern there.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He-like initialization. Good for ReLU networks.
        scale = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.weights = [
            [
                [
                    [random.gauss(0.0, scale) for _ in range(kernel_size)]
                    for _ in range(kernel_size)
                ]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]

        self.biases = [0.0 for _ in range(out_channels)]

    def forward(self, x):
        self.input = x
        self.padded_input = pad4d(x, self.padding)

        batch, channels, height, width = shape4(self.padded_input)
        kernel = self.kernel_size
        stride = self.stride

        out_h = (height - kernel) // stride + 1
        out_w = (width - kernel) // stride + 1

        out = zeros_4d(batch, self.out_channels, out_h, out_w)

        for b in range(batch):
            for f in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        total = self.biases[f]
                        row = i * stride
                        col = j * stride

                        for c in range(channels):
                            for ki in range(kernel):
                                for kj in range(kernel):
                                    image_value = self.padded_input[b][c][row + ki][col + kj]
                                    kernel_value = self.weights[f][c][ki][kj]
                                    total += image_value * kernel_value

                        out[b][f][i][j] = total

        self.output = out
        return out

    def backward(self, dout, learning_rate):
        batch, channels, height, width = shape4(self.padded_input)
        _, filters, out_h, out_w = shape4(dout)

        kernel = self.kernel_size
        stride = self.stride

        d_padded_input = zeros_4d(batch, channels, height, width)

        d_weights = [
            [
                [[0.0 for _ in range(kernel)] for _ in range(kernel)]
                for _ in range(channels)
            ]
            for _ in range(filters)
        ]

        d_biases = [0.0 for _ in range(filters)]

        for b in range(batch):
            for f in range(filters):
                for i in range(out_h):
                    for j in range(out_w):
                        grad = dout[b][f][i][j]
                        d_biases[f] += grad

                        row = i * stride
                        col = j * stride

                        for c in range(channels):
                            for ki in range(kernel):
                                for kj in range(kernel):
                                    d_weights[f][c][ki][kj] += self.padded_input[b][c][row + ki][col + kj] * grad
                                    d_padded_input[b][c][row + ki][col + kj] += self.weights[f][c][ki][kj] * grad

        for f in range(filters):
            self.biases[f] -= learning_rate * d_biases[f]
            for c in range(channels):
                for ki in range(kernel):
                    for kj in range(kernel):
                        self.weights[f][c][ki][kj] -= learning_rate * d_weights[f][c][ki][kj]

        return unpad4d(d_padded_input, self.padding)


class ReLU:
    """
    ReLU activation.

    Formula:
        if x > 0 -> keep x
        if x <= 0 -> make it 0

    Educational idea:
    ReLU removes negative responses and keeps active features.
    """

    def forward(self, x):
        self.input = x
        batch, channels, height, width = shape4(x)

        out = zeros_4d(batch, channels, height, width)
        self.mask = zeros_4d(batch, channels, height, width)

        for b in range(batch):
            for c in range(channels):
                for i in range(height):
                    for j in range(width):
                        if x[b][c][i][j] > 0.0:
                            out[b][c][i][j] = x[b][c][i][j]
                            self.mask[b][c][i][j] = 1.0

        self.output = out
        return out

    def backward(self, dout):
        batch, channels, height, width = shape4(dout)
        dx = zeros_4d(batch, channels, height, width)

        for b in range(batch):
            for c in range(channels):
                for i in range(height):
                    for j in range(width):
                        dx[b][c][i][j] = dout[b][c][i][j] * self.mask[b][c][i][j]

        return dx


class MaxPool2D:
    """
    Max Pooling layer.

    Educational idea:
    - Take a small block, for example 2x2.
    - Keep only the largest number.
    - This reduces size and keeps the strongest feature.
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.input = x

        batch, channels, height, width = shape4(x)
        pool = self.pool_size
        stride = self.stride

        out_h = (height - pool) // stride + 1
        out_w = (width - pool) // stride + 1

        out = zeros_4d(batch, channels, out_h, out_w)
        self.max_positions = {}

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        row = i * stride
                        col = j * stride

                        best = -10**9
                        best_position = (row, col)

                        for pi in range(pool):
                            for pj in range(pool):
                                value = x[b][c][row + pi][col + pj]
                                if value > best:
                                    best = value
                                    best_position = (row + pi, col + pj)

                        out[b][c][i][j] = best
                        self.max_positions[(b, c, i, j)] = best_position

        self.output = out
        return out

    def backward(self, dout):
        batch, channels, height, width = shape4(self.input)
        _, _, out_h, out_w = shape4(dout)

        dx = zeros_4d(batch, channels, height, width)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        row, col = self.max_positions[(b, c, i, j)]
                        dx[b][c][row][col] += dout[b][c][i][j]

        return dx


class Flatten:
    """
    Flatten layer.

    Educational idea:
    CNN feature maps are 2D matrices.
    Dense layer wants a 1D list.
    Flatten converts 2D maps into one long list.
    """

    def forward(self, x):
        self.input_shape = shape4(x)
        batch, channels, height, width = self.input_shape

        out = []

        for b in range(batch):
            row = []
            for c in range(channels):
                for i in range(height):
                    for j in range(width):
                        row.append(x[b][c][i][j])
            out.append(row)

        self.output = out
        return out

    def backward(self, dout):
        batch, channels, height, width = self.input_shape
        dx = zeros_4d(batch, channels, height, width)

        for b in range(batch):
            index = 0
            for c in range(channels):
                for i in range(height):
                    for j in range(width):
                        dx[b][c][i][j] = dout[b][index]
                        index += 1

        return dx


class Dense:
    """
    Fully connected layer.

    Educational idea:
    After CNN extracts features, Dense uses those features to make the final decision.
    """

    def __init__(self, in_features, out_features):
        scale = math.sqrt(2.0 / in_features)

        self.weights = [
            [random.gauss(0.0, scale) for _ in range(in_features)]
            for _ in range(out_features)
        ]

        self.biases = [0.0 for _ in range(out_features)]

    def forward(self, x):
        self.input = x
        batch = len(x)
        out_features = len(self.weights)
        in_features = len(self.weights[0])

        out = [[0.0 for _ in range(out_features)] for _ in range(batch)]

        for b in range(batch):
            for o in range(out_features):
                total = self.biases[o]
                for i in range(in_features):
                    total += x[b][i] * self.weights[o][i]
                out[b][o] = total

        self.output = out
        return out

    def backward(self, dout, learning_rate):
        batch = len(dout)
        out_features = len(self.weights)
        in_features = len(self.weights[0])

        dx = [[0.0 for _ in range(in_features)] for _ in range(batch)]

        d_weights = [
            [0.0 for _ in range(in_features)]
            for _ in range(out_features)
        ]

        d_biases = [0.0 for _ in range(out_features)]

        for b in range(batch):
            for o in range(out_features):
                grad = dout[b][o]
                d_biases[o] += grad

                for i in range(in_features):
                    d_weights[o][i] += self.input[b][i] * grad
                    dx[b][i] += self.weights[o][i] * grad

        for o in range(out_features):
            self.biases[o] -= learning_rate * d_biases[o]
            for i in range(in_features):
                self.weights[o][i] -= learning_rate * d_weights[o][i]

        return dx


class SoftmaxCrossEntropy:
    """
    Softmax + Cross Entropy.

    Softmax converts final scores into probabilities.
    Cross Entropy measures how wrong the prediction is.
    """

    def forward(self, logits, labels):
        self.probabilities = softmax(logits)
        self.labels = labels

        loss = 0.0
        for probs, label in zip(self.probabilities, labels):
            loss += -math.log(probs[label] + 1e-12)

        return loss / len(labels)

    def backward(self):
        batch = len(self.labels)
        grad = []

        for probs, label in zip(self.probabilities, self.labels):
            row = probs[:]
            row[label] -= 1.0
            row = [value / batch for value in row]
            grad.append(row)

        return grad


def softmax(logits):
    result = []

    for row in logits:
        maximum = max(row)
        exps = [math.exp(value - maximum) for value in row]
        total = sum(exps)
        result.append([value / total for value in exps])

    return result
