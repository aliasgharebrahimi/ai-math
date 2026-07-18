##🔍 1. What is linear algebra?

> Linear algebra is the study of vectors and matrices, providing the mathematical framework and language essential for artificial intelligence.

---

## 🤖 2. So, where exactly is it used in artificial intelligence?

As previously mentioned, the language of artificial intelligence is linear algebra (involving matrices, vectors, etc.), and it is utilized throughout the field. For instance:

* 🖼️ **Image Processing:** An image is fed into a model as a matrix.
* 🧠 **CNN Layers:** The kernel—which is itself a matrix—slides over the image.
* 🗺️ **Feature Maps:** The layer's output—a feature map—is also a matrix.
* 📉 **Jacobian Matrix:** Used for derivative (a concept we will also cover), which is also a matrix.

*Thus, it is evident that linear algebra is fundamental to every aspect of artificial intelligence.*

---



---

## 📚 Linear Algebra Concepts
### Basic Concepts

| Concept | Description | Mathematical Symbol |
| :--- | :--- | :--- |
| **Scalar** | A number used to scale vectors, representing magnitude without direction. | $c, \alpha, \lambda$ |
| **Vectors** | An ordered list of numbers representing direction and magnitude in a multidimensional space. | $\mathbf{v}, \mathbf{u}$ |
| **Matrices** | A rectangular array of numbers. Can represent linear transformations or collections of vectors. | $A, B$ |
| **Norm** | For a given vector, the distance from the point represented by that vector to the origin of the coordinate system is called the "Norm." | ‖v‖₁, ‖v‖₂, ‖v‖∞ |
| **Unit vector** | A vector is called a unit vector if its norm is exactly 1. | v̂ |
| **The angle between two vectors** | It is very simple: it refers to the angle formed by two vectors when drawn from the same origin point. | |
| **Gauss's method** | A method for solving a system of linear equations, known as Gaussian elimination. | |
| **Augmented matrix** | An augmented matrix is ​​a matrix used to solve a system of linear equations via the Gaussian method, making the solution process faster and more streamlined. | |
| **Reflection** | If the reflection is across the x-axis, it becomes (x, y) → (x, −y), and if it is across the y-axis, it becomes (x, y) → (−x, y). | |
| **Change the scale** | Multiplying a vector by a scalar allows for rescaling—that is, changing the size of the vector. | |
| **Shear** | This means changing its spatial shape without changing its area. | |
| **Scaling** | This implies a change in its area and, in some cases, a change in its spatial shape—naturally, taking proportionality into account. | |
| **90-degree rotation** | There is a simple rule for this: the new x-coordinate equals the negative of the old y-coordinate, and the new y-coordinate equals the old x-coordinate. Note that for a counter-clockwise rotation, a negative sign must be included; the formula is (x, y) → (-y, x). | (x, y) → (-y, x), (x, y) → (y, -x) |
| **180-degree turn** | In the form (x, y) → (-x, -y) | (x, y) → (-x, -y) |
| **270-degree turn** | In the form (x,y)→(y,−x), (x,y)→(-y,x) | (x,y)→(y,−x), (x,y)→(-y,x) |
| **Dot Product** | An operation between two vectors that returns a scalar, measuring their alignment. | $\mathbf{a} \cdot \mathbf{b}$ |
| **Matrix Multiplication** | An operation where the number of columns in the first matrix must match the rows in the second. | $AB$ |
| **Linear Combination** | The sum of scaled vectors: $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_n\mathbf{v}_n$. | $\sum_{i=1}^{n} c_i \mathbf{v}_i$ |
| **Span** | The set of all possible linear combinations of a given set of vectors. | $\text{span}(\mathcal{B})$ |
| **Linearly Independent** | A set where no vector can be formed by a linear combination of the others. | $\text{LI}(\mathcal{B})$ |
| **Linearly Dependent** | A set where at least one vector is a linear combination of the others. | $\text{LD}(\mathcal{B})$ |
| **Basis** | A set that is both linearly independent and spans the entire space. | $\text{span}(\mathcal{B})=V \land \text{LI}(\mathcal{B})$ |
| **Zero Matrix** | A matrix consisting entirely of zeros. | $\mathbf{0}$ |
| **Matrix Transpose** | Flipping a matrix over its diagonal, switching rows and columns. | $A^\top$ |
| **Identity Matrix** | A square matrix with 1s on the main diagonal and 0s elsewhere. |    I<sub>n<sub> |
| **Determinant** | In short, the determinant represents the magnitude of the volume change a matrix can produce; it can provide important information about a square matrix, such as whether it is invertible. | det(A) |
| **Matrix Inversion** | The inverse matrix $A^{-1}$ is a matrix that undoes the effect of the linear transformation produced by $A$, returning the space to its original state. This is possible only if the transformation is "non-singular"—meaning its determinant is non-zero ($\det(A) \neq 0$). | $A^{-1}$ |

### 🔢 Mathematical explanation
#### Presenting formulas and examining concepts mathematically.

<hr>

##### Scalar:
c = 6.2

<hr>

##### Vector:
v = [v₁, v₂, v₃] ,   u = [u₁, u₂, u₃]

##### Example:
v = [6.2, 5.3, 8.6]

<hr>

##### Scaling a Vector by a Scalar:
α · v = [αv₁, αv₂, ..., αvₙ]

##### Example:
α = 6.0  
v = [1.0, 2.0, 6.0]

z = α · v = [1.0 × 6.0, 2.0 × 6.0, 6.0 × 6.0]  
z = [6.0, 12.0, 36.0]

<hr>

##### Matrix:
A = [[a₁₁, a₁₂, a₁₃],  
     [a₂₁, a₂₂, a₂₃],  
     [a₃₁, a₃₂, a₃₃]]

##### Example:
B = [[2., 1., 6.],  
     [8., 9., 4.],  
     [7., 1., 6.]]

<hr>

---

## 💡 Important Notes

> [!NOTE]
> **In PyTorch**, we refer to these vectors, matrices, and so on as **"tensors"**—a vector is a 1D tensor, a matrix is a 2D tensor, and so forth (3D tensor, etc.).

> [!TIP]
> **In CNN layers:**
> * ⚪ **Black-and-white images:** Represented as a matrix or 2D tensor.
> * 🌈 **RGB (Color) images:** Represented as a 3D tensor (three matrices).

---

### 🤖 Alright, now that we understand what linear algebra is and where it's used, let's dive into the code and see it in action.