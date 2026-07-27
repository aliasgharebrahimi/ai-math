import torch
from typing import Tuple
import torch.nn.functional as F
from numpy.testing import print_coercion_tables
from torch._dynamo import side_effects

device = "cuda" if torch.cuda.is_available() else "cpu"
version = torch.__version__

class BasicCodes:

    def __init__(self, device, dtype, requires_grad):
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        self.scalar = torch.tensor(
            [6.2], 
            device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)  # torch.Size([1])

        # Vector
        # Of course, in PyTorch, row vectors and column vectors are not distinct in the way they are in mathematics.
        self.row_vector = torch.tensor(

            [6.2, 5.3, 8.6], 

            device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad
) # torch.Size([3]) 

        self.column_vector = torch.tensor(

            [7.0,
            12.3,
            1.0], 

            device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad # torch.Size([3])
)        
        # matrix
        self.matrix = torch.tensor(
            [
                [2., 1., 6.],
                [8., 9., 4.],
                [7., 1., 6.]

            ], 
            device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad  #
) # torch.Size([3, 3])
        
        # tensor 3d
        self.tensor = torch.tensor(
        [
            [[4., 4.],
             [7., 8.]],    
             
            [[7., 2.],
             [1., 1.]]
        ]
        ,device=self.device, 
        dtype=self.dtype, 
        requires_grad=self.requires_grad
)  # torch.Size([2, 2, 2])
        
        return self.scalar, self.row_vector, self.column_vector, self.matrix, self.tensor
        
    def norm(self) -> Tuple[torch.Tensor, ...]:
        # Norm
        self.vector_norm = torch.tensor(

            [1.0, 9.0]
            
            , device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)
        
        self.norm1 = torch.linalg.norm(self.vector_norm, ord=1) # tensor(10., grad_fn=<LinalgVectorNormBackward0>)
        self.norm2 = torch.linalg.norm(self.vector_norm, ord=2)  # tensor(9.0554, grad_fn=<LinalgVectorNormBackward0>)
        self.normb = torch.linalg.norm(self.vector_norm, ord=float("inf"))  # tensor(9., grad_fn=<LinalgVectorNormBackward0>)

        return self.norm1, self.norm2, self.normb

    def unit_vector(self) -> Tuple[torch.Tensor, ...]:
        # Two ways to understand a Unit Vector
        self.unit_vector1 = torch.tensor(
            
            [1.0, 0.0]

            , device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)
        self.on_unit_vector = torch.tensor(

            [8.0, 9.0]
            
            ,device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)

        # Method 1
        self.norm_unit = torch.norm(self.unit_vector1) # tensor(1.)
        self.norm_nounit = torch.norm(self.on_unit_vector) # tensor(12.0416)


        # Method 2
        self.iscloseT = torch.isclose(self.norm_unit, torch.tensor([1.0]), rtol=0.0001) # tensor([True])
        self.iscloseF = torch.isclose(self.norm_nounit, torch.tensor([1.0]), rtol=0.0001) # tensor([False])

        # Converting a vector to a unit vector
        self.conversion_unit_vector = F.normalize(self.on_unit_vector, p=2, dim=0)

        return self.norm_unit, self.norm_nounit, self.iscloseT, self.iscloseF, self.conversion_unit_vector

    def angle(self) -> torch.Tensor:

        """
        First, we set the vector's length to 1; having 1 in the denominator means the 
        result remains unchanged, which significantly reduces the amount of calculation required
        Determining the angle using a method that minimizes calculations.
        """

        # Definition of two vectors
        self.angle_v1 = torch.tensor(
            
            [1.0, 3.0]

            ,device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)
        self.angle_v2 = torch.tensor(

            [8.0, 7.0]

            ,device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)

        # Conversion to a unit vector
        self.angle_v1_norml = F.normalize(self.angle_v1, p=2, dim=0)
        self.angle_v2_norm2 = F.normalize(self.angle_v2, p=2, dim=0)

        # Computing the dot product
        # Here, since the norms of the vectors are 1, division is not necessary.
        
        self.cos_tehta_dot = torch.dot(self.angle_v1_norml, self.angle_v2_norm2)

        # Limiting the cosine to prevent errors
        self.cos_theta_clamp = torch.clamp(self.cos_tehta_dot, 1.0, -1.0)

        # angle
        self.angle1 = torch.arccos(self.cos_theta_clamp)

        # Converting radians to degrees
        self.angle_in_degrees = torch.rad2deg(self.angle1)

        return self.angle_in_degrees
    
    def gauss1(self) ->Tuple[torch.Tensor, ...]:

        # Constructing an augmented matrix
        self.a = torch.tensor(
            [   
                [2, 4],
                [4, 2]
            
            ]
            ,device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)
        self.b = torch.tensor(

            [12, 18]

            ,device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)
        self.z = torch.unsqueeze(self.b, dim=1)
        self.x = torch.cat([self.a, self.z], dim=1)

        # Solving by Gauss's method
        self.gauss = torch.linalg.solve(self.a, self.z)

        return self.x, self.gauss

    def reflection(self) -> Tuple[torch.Tensor, ...]:

        self.v = torch.tensor(

            [8., 9.],

            device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)

        self.rx = torch.tensor(
            [
                [1., 0.],
                [0., -1.]
            ],

        device=self.device,
        dtype=self.dtype,
        requires_grad=self.requires_grad)

        self.ry = torch.tensor(
            [
                [-1., 0.],
                [0., 1.]
            ],
        
        device=self.device,
        dtype=self.dtype,
        requires_grad=self.requires_grad)

        self.matmul1 = torch.matmul(self.rx, self.v)
        self.matmul2 = torch.matmul(self.ry, self.v)

        return self.matmul, self.matmul2

    def shear(self) -> Tuple[torch.Tensor, ...]:

        self.u = torch.tensor(

            [9., 2.],

            device=self.device, 
            dtype=self.dtype, 
            requires_grad=self.requires_grad)
        self.k = 2.0

        self.shx = torch.tensor(
            [
                [1., self.k],
                [0., 1.]
            ],
        
        device=self.device,
        dtype=self.dtype,
        requires_grad=self.requires_grad)

        self.shy = torch.tensor(
                    [
                        [1., 0.],
                        [self.k, 1.]
                    ],
                
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.requires_grad)

        self.mat1 = torch.matmul(self.shx, self.u)
        self.mat2 = torch.matmul(self.shy, self.u)

        return self.mat1, self.mat2

    def scaling(self) -> Tuple[torch.Tensor, ...]:

        self.v = torch.tensor(

            [6.0, 1.0],

        device = self.device,
        dtype = self.dtype,
        requires_grad = self.requires_grad)
        self.sx = 2.0
        self.sy = 2.0

        self.s = torch.tensor(
            [
                [self.sx, 0.],
                [0., self.sy]
            ]
        ,device = self.device,
        dtype = self.dtype,
        requires_grad = self.requires_grad)

        self.sca = torch.matmul(self.s, self.v)

        return self.sca

    def ro90(self) -> Tuple[torch.Tensor, ...]:

        self.v = torch.tensor(

            [1.0, 2.0]

        ,device = self.device,
        dtype = self.dtype,
        requires_grad = self.requires_grad)
        #self.v1 = torch.unsqueeze(self.v, dim=1)

        self.r = torch.tensor(
            [
                [0., 1.],
                [-1., 0.]
            ]
        ,device = self.device,
        dtype = self.dtype,
        requires_grad = self.requires_grad)

        self.r_ = torch.tensor(
            [
                [0., -1.],
                [1., 0.]
            ]
        ,device = self.device,
        dtype = self.dtype,
        requires_grad = self.requires_grad)

        self.ro = torch.matmul(self.r, self.v)
        self.ro_ = torch.matmul(self.r_,self.v)

        return self.ro, self.ro_

    def  ro180(self) -> Tuple[torch.Tensor, ...]:

        self.v = torch.tensor(

            [8.0, 8.0]

        , device=self.device,
        dtype=self.dtype,
        requires_grad=self.requires_grad)

        self.r = torch.tensor(
            [
                [-1., 0.],
                [0., -1.]
            ]
        , device=self.device,
        dtype=self.dtype,
        requires_grad=self.requires_grad)

        self.ro = torch.matmul(self.r, self.v)

        return self.ro

    def dot(self) -> Tuple[torch.Tensor, ...]:

        self.v = torch.tensor(

            [6., 7.]

            , device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad)
        self.v2 = torch.tensor(

            [1., 9.]

            , device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad)

        self.dot = torch.dot(self.v, self.v2)

        return self.dot


    def matmul(self) -> torch.Tensor:

        self.mt1 = torch.tensor(
          [
            [1., 9., 3.],
            [4., 4., 3.],
            [7., 7., 0.]
        ]
        , device=self.device,
        dtype=self.dtype,
        requires_grad=self.requires_grad)

        self.mt2 = torch.tensor(
        [
            [4., 9., 2.],
            [1., 2., 3.],
            [9., 6., 1.]
        ]
        , device=self.device,
        dtype=self.dtype,
        requires_grad=self.requires_grad)

        self.matmul_res = torch.matmul(self.mt1, self.mt2)

        return self.matmul_res


object_BasicCodes = BasicCodes(device=device, dtype=torch.float32, requires_grad=False)

def main():

    print("="*50)
    print("PYTORCH BASIC CONCEPTS DEMONSTRATION")
    print(f"PyTorch Version: {version}")
    print(f"System device: {device}")
    print("="*50)

    # 1. نمایش تنسورها
    print(60*"=")
    print("[1] Tensors & Structures:")
    s, rv, cv, m, t3 = object_BasicCodes.tensors()
    print(f" - Scalar:    {s} | Shape: {s.shape}")
    print(f" - Row Vector: {rv} | Shape: {rv.shape}")
    print(f" - Col Vector: {cv} | Shape: {cv.shape}")
    print("\n")
    print(f" - Matrix:     \n{m} | Shape: {m.shape}")
    print("\n")
    print(f" - 3D Tensor:  \n{t3} | Shape: {t3.shape}")

    # 2. نمایش نرم‌ها
    print(60 * "=")
    print("[2] Norm Calculations:")
    n1, n2, nb = object_BasicCodes.norm()
    print(f" - L1 Norm:    {n1.item():.4f}")
    print(f" - L2 Norm:    {n2.item():.4f}")
    print(f" - L-inf Norm: {nb.item():.4f}")

    # 3. نمایش بردار واحد
    print(60 * "=")
    print("[3] Unit Vector Check:")
    q, q2, is_u, is_not_u, c = object_BasicCodes.unit_vector()
    a = object_BasicCodes.on_unit_vector
    print(f" - Vector u norm: {q.item():.2f} | Is Unit? {is_u.item()}")
    print(f" - Vector v norm: {q2.item():.2f} | Is Unit? {is_not_u.item()}")
    print(f" - vector {a} unit: {c}")

    # 4. angle 2vector
    print(60 * "=")
    print("[4] angle 2vector:")
    angle = object_BasicCodes.angle()
    print(f"angle 2vector: {angle}")

    # 5. gauss_equation
    print(60 * "=")
    p, x = object_BasicCodes.gauss1()
    print("[5] gauss:")
    print(f" - matrix plugin:{p}")
    print(f" - x:{x[0]}, y:{x[1].detach()}")

    # 6. Reflection
    print(60 * "=")
    m1, m2 = object_BasicCodes.reflection()
    print("[6] Reflection:")
    print(f" - a Reflection x:{m1}")
    print(f" - a Reflection y:{m2}")

    # 7. Shear
    print(60 * "=")
    ms1, ms2 = object_BasicCodes.shear()
    print("[7] Shear:")
    print(f" - a Shear x:{ms1}")
    print(f" - a Shear y:{ms2}")

    # 8. Scaling
    print(60 * "=")
    sca = object_BasicCodes.scaling()
    print("[8] Scaling:")
    print(f" - a Scaling vector: {sca}")

    # Rotation 90
    print(60 * "=")
    ro, ro_ = object_BasicCodes.ro90()
    print("[9] Rotation 90:")
    print(f" - a Rotation 90:{ro}")
    print(f" - a Rotation -90:{ro_}")

    # Rotation 180
    print(60 * "=")
    ro2 = object_BasicCodes.ro180()
    print("[10] Rotation 180:")
    print(f" - a Rotation 180:{ro2}")

    # Dot Product
    print(60 * "=")
    dot_product = object_BasicCodes.dot()
    print("[11] Dot Product:")
    print(f" - a Dot Product: {dot_product}")

    # matmul matrix
    print(60 * "=")
    matmul_matrix = object_BasicCodes.matmul()
    print(f"[12] Matmul Matrix: {matmul_matrix}")


if __name__ == "__main__":

    main()

