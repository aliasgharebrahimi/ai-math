import torch

device = "cpu"
# Scalar
scalar = torch.tensor([6.2], device=device, dtype=torch.float16)  # torch.Size([1])

# Vector
# Of course, in PyTorch, row vectors and column vectors are not distinct in the way they are in mathematics.
row_vector = torch.tensor([6.2, 5.3, 8.6], device=device, dtype=torch.float16) # torch.Size([3]) 

column_vector = torch.tensor([7.0,
                              12.3,
                              1.0], device=device, dtype=torch.float16) # torch.Size([3])

# matrix
matrix = torch.tensor([[2., 1., 6.],
                       [8., 9., 4.],
                       [7., 1., 6.]], device=device, dtype=torch.float16)  # torch.Size([3, 3])

# tensor 3d
tensor = torch.tensor([[[4., 4.],
                       [7., 8.]],
                       
                       [[7., 2.],
                       [1., 1.]]], device=device, dtype=torch.float16)  # torch.Size([2, 2, 2])

# Norm
vector_norm = torch.tensor([1.0, 9.0], device=device, dtype=torch.float16)
norm1 = torch.linalg.norm(vector_norm, ord=1) # tensor(10., grad_fn=<LinalgVectorNormBackward0>)
norm2 = torch.linalg.norm(vector_norm, ord=2)  # tensor(9.0554, grad_fn=<LinalgVectorNormBackward0>)
normb = torch.linalg.norm(vector_norm, ord=float("inf"))  # tensor(9., grad_fn=<LinalgVectorNormBackward0>)

# Two ways to understand a Unit Vector
u = torch.tensor([1.0, 0.0], device=device, dtype=torch.float16)
v = torch.tensor([8.0, 9.0], device=device, dtype=torch.float16)

# Method 1
q = torch.norm(u) # tensor(1.)
q2 = torch.norm(v) # tensor(12.0416)


# Method 2
i = torch.isclose(q, torch.tensor([1.0]), rtol=0.0001) # tensor([True])
i2 = torch.isclose(q2, torch.tensor([1.0]), rtol=0.0001) # tensor([False])