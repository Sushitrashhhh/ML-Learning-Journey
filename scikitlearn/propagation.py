import numpy as np

#sigmoid activation + derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#input dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
#output dataset
y = np.array([[0],[1],[1],[0]])

#random seeding
np.random.seed(42)

#initialize weights and biases
input_neurons=2
hidden_neurons=2
output_neurons=1

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

#learning rate
lr=0.1 #learning rate
epochs=10000

# training loop
for epoch in range(epochs):
    #forawrd prop
    z1 = np.dot(X,W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1,W2) + b2
    a2 = sigmoid(z2)

    #loss(MSE)
    loss = np.mean((y - a2) ** 2)

    #backprop
    #output layer
    d_a2 = (a2-y)*sigmoid_derivative(a2)
    #hidden layer
    d_a1 = d_a2.dot(W2.T) * sigmoid_derivative(a1)

    #gradient for wieghts and biases
    W2-=lr*a1.T.dot(d_a2)
    b2-=lr*np.sum(d_a2, axis=0, keepdims=True)

    W1-=lr*X.T.dot(d_a1)
    b1-=lr*np.sum(d_a1, axis=0, keepdims=True)

    #print loss every 2000 epochs
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

#final output after training
print("Final output after training:", a2)
'''
```
1. Imports and Activation Functions
import numpy as np: Imports NumPy for numerical operations.
sigmoid(x): Implements the sigmoid activation function.
sigmoid_derivative(x): Computes the derivative of the sigmoid function (used in backpropagation).

2. Dataset
X: Input data for XOR problem (4 samples, 2 features each).
y: Output labels for XOR (4 samples, 1 output each).

3. Initialization
Random seed: Ensures reproducibility.
Network structure: 2 input neurons, 2 hidden neurons, 1 output neuron.
Weights (W1, W2): Randomly initialized for input→hidden and hidden→output layers.
Biases (b1, b2): Initialized to zeros.

4. Training Loop
Epochs: 10,000 iterations.
Forward Propagation:
z1 = X·W1 + b1: Linear combination for hidden layer.
a1 = sigmoid(z1): Activation for hidden layer.
z2 = a1·W2 + b2: Linear combination for output layer.
a2 = sigmoid(z2): Activation for output layer (final prediction).

Loss Calculation:
Mean squared error between predictions (a2) and true labels (y).
Backpropagation:
Compute gradients for output and hidden layers using the chain rule and sigmoid derivative.
Update weights and biases using gradient descent.

Progress Print:
Every 2000 epochs, prints the current loss.
5. Final Output
After training, prints the network’s output for all inputs.

Summary:
This code implements a simple neural network from scratch (no frameworks) to solve the XOR problem using one hidden layer, sigmoid activations, and manual backpropagation. It demonstrates the core mechanics of neural network training.
```
'''