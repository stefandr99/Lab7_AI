import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


inputs = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

real_outputs = np.array([[0, 1, 1, 0]]).T

hidden_weights = 2 * np.random.random((2, 2)) - 1
output_weights = 2 * np.random.random((2, 1)) - 1
rate = 1


for epoch in range(10000):
    hidden_layer = sigmoid(np.dot(inputs, hidden_weights))
    outputs_generated = sigmoid(np.dot(hidden_layer, output_weights))

    error = real_outputs - outputs_generated
    adjustments_output = error * sigmoid_derivative(outputs_generated)
    output_weights += rate * np.dot(hidden_layer.T, adjustments_output)

    error2 = adjustments_output.dot(output_weights.T)
    adjustments_hidden = error2 * sigmoid_derivative(hidden_layer)
    hidden_weights += rate * np.dot(inputs.T, adjustments_hidden)



print("Output layer")
print(outputs_generated)



