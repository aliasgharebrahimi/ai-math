import torch

device = "cpu"
# Scalar
scalar = torch.tensor([6.2], dtype=torch.float16, device=device, requires_grad=True)

# Vector
# Of course, in PyTorch, row vectors and column vectors are not distinct in the way they are in mathematics.
row_vector = torch.tensor([6.2, 5.3, 8.6], dtype=torch.float32, device=device, requires_grad=True)
column_vector = torch.tensor([7.0,
                              12.3,
                              1.0], dtype=torch.float32, device=device, requires_grad=True)

# matrix
matrix = torch.tensor([[2., 1., 6.],
                       [8., 9., 4.],
                       [7., 1., 6.]], dtype=torch.float32, device=device, requires_grad=True)

# tensor 3d
tensor = torch.tensor([[[4., 4.],
                       [7., 8.]],
                       
                       [[7., 2.],
                       [1., 1.]]])

# Norm
norm1 = torch.linalg.norm(row_vector, ord=1)
norm2 = torch.linalg.norm(row_vector, ord=2)
normb = torch.linalg.norm(row_vector, ord=float("inf"))

# Two ways to understand a Unit Vector
u = torch.tensor([1.0, 0.0], device=device)
v = torch.tensor([8.0, 9.0], device=device)

# Method 1
q = torch.norm(v)
q2 = torch.norm(u)

# Method 2
i = torch.isclose(q, torch.tensor([1.0]))
i2 = torch.isclose(q2, torch.tensor([1.0]))

print(scalar.shape) # torch.Size([1])
print(row_vector.shape, column_vector.shape) # torch.Size([3]) torch.Size([3])
print(matrix.shape) # torch.Size([3, 3])
print(tensor.shape) # torch.Size([2, 2, 2])
print(norm1) # tensor(20.1000, grad_fn=<LinalgVectorNormBackward0>)
print(norm2) # tensor(11.8528, grad_fn=<LinalgVectorNormBackward0>)
print(normb) # tensor(8.6000, grad_fn=<LinalgVectorNormBackward0>)
print(q, q2) # tensor(1.), tensor(12.0416)
print(i, i2) # tensor([False]) tensor([True])