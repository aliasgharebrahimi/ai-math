import torch
from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"
version = torch.__version__

# Scalar

class BasicCodes:

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        scalar = torch.tensor([6.2], device=self.device, dtype=self.dtype)  # torch.Size([1])

        # Vector
        # Of course, in PyTorch, row vectors and column vectors are not distinct in the way they are in mathematics.
        row_vector = torch.tensor([6.2, 5.3, 8.6], device=self.device, dtype=self.dtype) # torch.Size([3]) 

        column_vector = torch.tensor([7.0,
                                      12.3,
                                      1.0], device=self.device, dtype=self.dtype) # torch.Size([3])
        
        # matrix
        matrix = torch.tensor([[2., 1., 6.],
                               [8., 9., 4.],
                               [7., 1., 6.]], device=self.device, dtype=self.dtype)  # torch.Size([3, 3])

        # tensor 3d
        tensor = torch.tensor([[[4., 4.],
                                [7., 8.]],
                       
                                [[7., 2.],
                                 [1., 1.]]], device=self.device, dtype=self.dtype)  # torch.Size([2, 2, 2])
        
        return scalar, row_vector, column_vector, matrix, tensor
        
    def norm(self) -> Tuple[torch.Tensor, ...]:
        # Norm
        vector_norm = torch.tensor([1.0, 9.0], device=self.device, dtype=self.dtype)
        norm1 = torch.linalg.norm(vector_norm, ord=1) # tensor(10., grad_fn=<LinalgVectorNormBackward0>)
        norm2 = torch.linalg.norm(vector_norm, ord=2)  # tensor(9.0554, grad_fn=<LinalgVectorNormBackward0>)
        normb = torch.linalg.norm(vector_norm, ord=float("inf"))  # tensor(9., grad_fn=<LinalgVectorNormBackward0>)

        return norm1, norm2, normb

    def unit_vector(self) -> Tuple[torch.Tensor, ...]:
        # Two ways to understand a Unit Vector
        u = torch.tensor([1.0, 0.0], device=self.device, dtype=self.dtype)
        v = torch.tensor([8.0, 9.0], device=self.device, dtype=self.dtype)

        # Method 1
        q = torch.norm(u) # tensor(1.)
        q2 = torch.norm(v) # tensor(12.0416)


        # Method 2
        i = torch.isclose(q, torch.tensor([1.0]), rtol=0.0001) # tensor([True])
        i2 = torch.isclose(q2, torch.tensor([1.0]), rtol=0.0001) # tensor([False])

        return q, q2, i, i2

objct_BasicCodes = BasicCodes(device=device, dtype=torch.float32)

def main():

    print("="*50)
    print("PYTORCH BASIC CONCEPTS DEMONSTRATION")
    print(f"PyTorch Version: {version}")
    print("="*50)

    # 1. نمایش تنسورها
    print("\n[1] Tensors & Structures:")
    s, rv, cv, m, t3 = objct_BasicCodes.tensors()
    print(f" - Scalar:    {s} | Shape: {s.shape}")
    print(f" - Row Vector: {rv} | Shape: {rv.shape}")
    print(f" - Col Vector: {cv} | Shape: {cv.shape}")
    print(f" - Matrix:     \n{m} | Shape: {m.shape}")
    print(f" - 3D Tensor:  \n{t3} | Shape: {t3.shape}")

    # 2. نمایش نرم‌ها
    print("\n[2] Norm Calculations:")
    n1, n2, nb = objct_BasicCodes.norm()
    print(f" - L1 Norm:    {n1.item():.4f}")
    print(f" - L2 Norm:    {n2.item():.4f}")
    print(f" - L-inf Norm: {nb.item():.4f}")

    # 3. نمایش بردار واحد
    print("\n[3] Unit Vector Check:")
    q, q2, is_u, is_not_u = objct_BasicCodes.unit_vector()
    print(f" - Vector u norm: {q.item():.2f} | Is Unit? {is_u.item()}")
    print(f" - Vector v norm: {q2.item():.2f} | Is Unit? {is_not_u.item()}")

    print("\n" + "="*50)

if __name__ == "__main__":

    main()