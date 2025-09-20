import torch

# 1. Create a PyTorch tensor instead of a NumPy array.
#    dtype=torch.float32 is standard for deep learning.
neuron_outputs = torch.tensor([-10, -2, 0, 1.5, 25], dtype=torch.float32)

# 2. Use the highly optimized built-in PyTorch ReLU function.
activation_function = torch.relu(neuron_outputs)

print(f'Original tensor: {neuron_outputs}')
print(f'Activated tensor: {activation_function}')