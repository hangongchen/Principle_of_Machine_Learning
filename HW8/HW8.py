import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def initialize_weights(layers, p=0.5):
    weights = []
    for i in range(len(layers) - 1):
        weights.append(np.random.uniform(0, p, (layers[i + 1], layers[i])))
    return weights

def forward_pass(x, weights):
    a = [x]
    z = []
    for theta in weights[:-1]:
        z_l = theta @ a[-1]
        z.append(z_l)
        a.append(sigmoid(z_l))
    z_l = weights[-1] @ a[-1]
    z.append(z_l)
    a.append(z_l)
    return a, z

def backward_pass(x, y, weights, a, z):
    gradients = [np.zeros_like(w) for w in weights]
    delta = a[-1] - y
    gradients[-1] = delta @ a[-2].T
    for l in range(len(weights) - 2, -1, -1):
        delta = (weights[l + 1].T @ delta) * sigmoid_derivative(z[l])
        gradients[l] = delta @ a[l].T
    return gradients

def update_weights(weights, gradients, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients[i]

def train_network(x, y, layers, p=0.5, learning_rate=0.05, iterations=50):
    weights = initialize_weights(layers, p)
    loss_history = []
    for _ in range(iterations):
        a, z = forward_pass(x, weights)
        gradients = backward_pass(x, y, weights, a, z)
        update_weights(weights, gradients, learning_rate)
        loss = 0.5*(y - a[-1])**2
        loss_history.append(loss.item())
    return loss_history, weights

x = np.array([[2], [1]])
y = 3
layers = [2, 2, 1]

loss_history, weights = train_network(x, y, layers, p=0.5, learning_rate=0.05, iterations=50)

plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Squared Error Loss')
plt.title('Training Loss Curve')
plt.show()

print(loss_history[-1])
