import numpy as np
from hw1 import NeuralNetwork

np.random.seed(42)
X = np.random.randn(2, 4)
Y = np.array([[1, 0, 0], [0, 1, 0]])

model = NeuralNetwork(4, 5, 3)
Z2, cache = model.forward(X)
loss = model.compute_loss(model.softmax(Z2), Y)
grads = model.backward(cache, Y)

# Numerical gradient for W2
eps = 1e-5
dW2_num = np.zeros_like(model.W2)
for i in range(model.W2.shape[0]):
    for j in range(model.W2.shape[1]):
        model.W2[i, j] += eps
        Z2_plus, _ = model.forward(X)
        loss_plus = model.compute_loss(model.softmax(Z2_plus), Y)
        
        model.W2[i, j] -= 2*eps
        Z2_minus, _ = model.forward(X)
        loss_minus = model.compute_loss(model.softmax(Z2_minus), Y)
        
        model.W2[i, j] += eps
        dW2_num[i, j] = (loss_plus - loss_minus) / (2*eps)

print("Max diff dW2:", np.max(np.abs(grads['W2'] - dW2_num)))

# Numerical gradient for W1
dW1_num = np.zeros_like(model.W1)
for i in range(model.W1.shape[0]):
    for j in range(model.W1.shape[1]):
        model.W1[i, j] += eps
        Z2_plus, _ = model.forward(X)
        loss_plus = model.compute_loss(model.softmax(Z2_plus), Y)
        
        model.W1[i, j] -= 2*eps
        Z2_minus, _ = model.forward(X)
        loss_minus = model.compute_loss(model.softmax(Z2_minus), Y)
        
        model.W1[i, j] += eps
        dW1_num[i, j] = (loss_plus - loss_minus) / (2*eps)

print("Max diff dW1:", np.max(np.abs(grads['W1'] - dW1_num)))
