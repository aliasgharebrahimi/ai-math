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
| **The angle between two vectors** | It is very simple: it refers to the angle formed by two vectors when drawn from the same origin point. | cos(θ) = (a · b) / ( ‖a‖ . ‖b‖ ) |
| **Gauss's method** | A method for solving a system of linear equations, known as Gaussian elimination. | R_i ← R_i - (a_i,j / a_p,j) × R_p |
| **Augmented matrix** | An augmented matrix is ​​a matrix used to solve a system of linear equations via the Gaussian method, making the solution process faster and more streamlined. | - |
| **Reflection** | If the reflection is across the x-axis, it becomes (x, y) → (x, −y), and if it is across the y-axis, it becomes (x, y) → (−x, y). | Rv |
| **Change the scale** | Multiplying a vector by a scalar allows for rescaling—that is, changing the size of the vector. | a . v |
| **Shear** | This means changing its spatial shape without changing its area. | v' |
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

# Linear Algebra Fundamentals

# Scalar:

`c = 6.2`

---

# Vector:

`v = [v₁, v₂, v₃]` , `u = [u₁, u₂, u₃]`

#### Example:

`v = [6.2, 5.3, 8.6]`

---

#### Scaling a Vector by a Scalar:

`α · v = [αv₁, αv₂, ..., αvₙ]`

#### Example:

`α = 6.0`  
`v = [1.0, 2.0, 6.0]`

`z = α · v = [1.0 × 6.0, 2.0 × 6.0, 6.0 × 6.0]`  
`z = [6.0, 12.0, 36.0]`

---

# Matrix:

`A = [a₁₁, a₁₂, a₁₃],`  
`    [a₂₁, a₂₂, a₂₃],`  
`    [a₃₁, a₃₂, a₃₃]`

#### Example:

`B = [2., 1., 6.],`  
`    [8., 9., 4.],`  
`    [7., 1., 6.]`

---

# Norm

Given a vector `v = [v₁, v₂, ..., vₙ]`, the different norms are defined as:

1. **L1 Norm (Manhattan Norm):**
   `∑ |vᵢ|`

2. **L2 Norm (Euclidean Norm):**
   `√(∑ |vᵢ|²)`

3. **L-infinity Norm (Max Norm):**
   `max(|vᵢ|)`

#### Example:

**Norm1:**

`u = [4.0, -3.0]`  
`n = 4.0 + 3.0`  
`n = 7.0`

**Norm2:**

`v = [4.0, -3.0]`  
`w = √(4.0² + 3.0²)`  
`w = √(16.0 + 9.0)`  
`w = √(25.0)`  
`w = 5.0`

**L-infinity Norm:**

`v = [4.0, -3.0]`  
`q = max(v)`  
`q = 4.0`

---

# Unit Vector

`v̂`

#### Example:

`v = [1.0, 0.0]`  
`p = 1.0 + 0.0`  
`p = 1.0`  

#### Example:
Converting a Norm vector to a unit vector:

`v̂ = v / ||v||`

#### Example:

`v = [4.0, -3.0]`  
`x = 4.0 + 3.0`  
`x = 7.0`  
`v̂ = v / x`  
`v̂ = [4.0, -3.0] / 7.0`  
`v̂ = [0.5714285714285714, 0.4285714285714286]`

#### Test:

`d = 0.5714285714285714 + 0.4285714285714286`  
`d = 1.0`

---

# The angle between two vectors:

To obtain it, we use the cosine.

The formula for obtaining it:

`cos(theta) = (a . b) / (|a| * |b|)`

#### Example:

`a = [8, 6, 1]`
`b = [4, 3, 3]`

**Dot Product:**

`z = a . b = [8, 6, 1] . [4, 3, 3] = [32, 18, 3] = [32 + 18 + 3] = 53`

**Norm a, b:**

`n1 = √(8² + 6² + 1²)`
`n2 = √(4² + 3² + 3²)`

`n1 = √(101)`
`n2 = √(34)`

`n1 = 10.04987562112089`
`n2 = 5.8309518948453`

**Paste into the formula:**

`cos(theta) = z / (n1 * n2) = 53 / (10.04987562112089 * 5.8309518948453)`

`cos(theta) = 53 / 58.60034129593444`

`cos(theta) = 0.9044315925115102`

So, we have found the cosine of the angle; this value indicates the degree of alignment between the two lines. First, we need to convert it into degrees using the arccos function:

`theta_radian = arccos(cos(theta))`

`theta_radian = 0.4405103241142635`

**Conversion to degrees:**

`Degree = 0.4405103241142635 * 180 / π`

`Degree ≈ 25.24`

---

# Solving a system of linear equations using the Gaussian method and an augmented matrix:

1. **Augmented Matrix Construction:** Convert the system of linear equations into an augmented matrix format.
2. **Forward Elimination (Row Operations):** Perform row operations to transform the matrix into **Row Echelon Form** (making all elements below the main diagonal equal to zero).
3. **Back Substitution:** Convert the simplified matrix back into equations and solve for the variables starting from the last one.

**Formula**

`R_i ← R_i - (a_i,j / a_p,j) × R_p`

**Explanation of the formula**

Divide the element `a_i,j` (located below the main diagonal) by `a_p,j` (the element in the same column in the row above). Multiply the result by the upper row `R_p`. Subtract this product from the lower row `R_i` and replace the lower row with the new values.

**Example:**

**System of equations:**

`2x + 4y = 12`
`4x + 2y = 18`

**Convert to augmented matrix:**

`[2 4 | 12],`
`[4 2 | 18]`

**Divide the number below the diagonal by the number above it in the same column:**

`4 / 2 = 2`

**Multiply by the top row:**

`2 . [2 4 | 12] = [4 8 | 24]`

**Subtract the obtained row from the row below it:**

`[4 8 | 24] - [4 2 | 18] = [0 6 | 6]`

**Plugin Matrix Update:**

`[2 4 | 12]`
`[0 6 | 6]`

**Solving an equation:**

`2x + 4y = 12`
`0x + 6y = 6`

`x = 4`
`y = 1`

# Reflection

`Rv`

**Its formula is as follows:**

**x-axis**

`(x, y) → (x, −y)`

**y-axis**

`(x, y) → (-x, y)`

**Example:**

`v = [8., 9.]`

**Reflection across the x-axis:**

`[8., 9.] → [8., -9.]`

**Reflection across the y-axis:*

`[8., 9.] → [-8., 9.]`

**Mathematically:**

`R_x = [1, 0],`
    `[0, -1]`

`R_y = [-1, 0],`
    `[0, 1]`

`z = R_xv = [8., -9.]`

`z2 = R_yv = [-8., 9.]`

# Shear

**Formula:**

**Shear on the x-axis:**

`x' = x + k × y`
`y' = y`

**Shear on the y-axis:**

`x' = x`
`y' = y + k × x`

**Example**

`v = [9.0, 2.0]`

**x:**

`k = 2`

`x' = 9.0 + 2.0 . 2.0`
`x' = 13.0`

`y' = 2.0`

`v' = [13.0, 2.0]`

**y:**

`v = [9.0, 2.0]`

`k = 2`

`y' = 2.0 + 2.0 . 9.0`
`y = 20.0`

`x' = 9.0`

`v' = [9.0, 20.0]`

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