import torch

a = torch.tensor([
    [8., 6., 2.],
    [3., 2., 0.],
    [9., 9., 9.]
])
b = torch.tensor([1., 3., 9.])
b = torch.unsqueeze(b, dim=0)

cat = torch.cat([a, b], dim=0)
print(cat)