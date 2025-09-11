def count_param(n_input, hidden_layers, n_output):
    """
    Count the number of parameters in a feedforward neural network.

    Parameters:
    n_input (int): Number of input features.
    hidden_layers (list of int): List containing the number of neurons in each hidden layer.
    n_output (int): Number of output neurons.

    Returns:
    int: Total number of parameters (weights + biases) in the network.
    """
    total_params = 0
    layer = [n_input] + hidden_layers + [n_output]

    for i in range(len(layer)-1):
        n_in = layer[i]
        n_out = layer[i+1]
        total_params += n_in * n_out + n_out  # weights + biases

    return total_params


print("linear model params:", count_param(3, [], 1))  
# 3*1 (weights) + 1 (bias) = 4

print("model with hidden layers: ", count_param(3, [4, 5], 1))  
# (3*4 + 4) + (4*5 + 5) + (5*1 + 1) = 47