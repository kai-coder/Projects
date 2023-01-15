import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import vertical_data, spiral_data
import math
from PIL import Image

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dinputs = np.dot(dvalues, self.weights.T)

        # Think of np.dot(dvalues.T, self.inputs).T
        # Calculates for each input which returns d_weight
        #   0 1 2      dot      0i 1i 2i 3i
        # [ 1 1 1 ]           [ 1  2  3  4 ]
        # [ 2 2 2 ]           [ 1  2  3  4 ]
        # [ 3 3 3 ]           [ 1  2  3  4 ]
        # d_weight[0][0] = 1 * 1 + 2 * 1 + 3 * 1
        # d_weight[0][1] = 1 * 2 + 2 * 2 + 3 * 2
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        self.negative_log_likelihoods = -np.log(correct_confidences)
        print(self.negative_log_likelihoods)
        return self.negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

X, y = spiral_data(samples=1, classes=1)

dense1 = Layer_Dense(2, 64)
activaton1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3)
activaton2 = Activation_ReLU()

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD()
lossStorage = []
accStorage = []
for i in range(1):
    dense1.forward(X)
    activaton1.forward(dense1.output)

    dense2.forward(activaton1.output)
    activaton2.forward(dense2.output)

    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    lossStorage.append(loss)
    accStorage.append(accuracy)
    if not i % 1000:
        print(f'epoch: {i}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activaton1.backward(dense2.dinputs)
    dense1.backward(activaton1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

# width = 300
# height = 300
# img = Image.new('RGB', (width, height))
# pixels = []
# for j in range(height):
#     for i in range(width):
#         dense1.forward(np.array([(i / width) * 2 - 1, -(j / height) * 2 + 1]))
#         activaton1.forward(dense1.output)
#
#         dense2.forward(activaton1.output)
#
#         activaton2.forward(dense2.output)
#
#         loss = loss_activation.forward(dense2.output, y)
#         pixels.append((int(loss_activation.output[0][1] * 255), int(loss_activation.output[0][2] * 255), int(loss_activation.output[0][0] * 255)))
# img.putdata(pixels)
# img.save('image.png')
# pixels = mpimg.imread("image.png")
# plt.imshow(pixels, extent=[-1, 1, -1, 1])
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", edgecolors='black')
# plt.show()
# plt.plot(range(101), lossStorage)
# plt.plot(range(101), accStorage)
# plt.show()
