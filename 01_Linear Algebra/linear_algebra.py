# Imports
import torch

# Defining a device to make the code more portable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Definition of a vector
tensor_vector = torch.tensor([6.0, 12.0, -37.0], dtype=torch.float32, device=device)

# Definition of a matrix
tensor_matrix = torch.tensor([[0.2, 6.0, -9.0],
                              [4.0, 8.0, 11.0],
                              [7.0, 1.0, -9.0]], dtype=torch.float32, device=device)

# Definition of a tensor 3d
tensor_3d = torch.tensor([[[0.2, 6.0, -9.0],
                           [4.0, 8.0, 11.0],
                           [7.0, 1.0, -9.0]],
 
                          [[0.2, 6.0, -9.0],
                           [4.0, 8.0, 11.0],
                           [7.0, 1.0, -9.0]]], dtype=torch.float32, device=device)

print(tensor_vector.shape) # torch.Size([3])
print(tensor_matrix.shape) # torch.Size([3, 3])
print(tensor_3d.shape) # torch.Size([2, 3, 3])