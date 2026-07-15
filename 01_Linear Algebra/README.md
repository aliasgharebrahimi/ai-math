# 📐 Linear Algebra

<img src="../Project photos/Photos in folder 01/Linear Algebra Visualization.jpg" alt="" />

---

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

| Concept | Description | Mathematical Symbol |
| :--- | :--- | :--- |
| **Scalar** | A number used to scale vectors, representing magnitude without direction. | $c, \alpha, \lambda$ |
| **Vectors** | An ordered list of numbers representing direction and magnitude in a multidimensional space. | $\mathbf{v}, \mathbf{u}$ |
| **Matrices** | A rectangular array of numbers. Can represent linear transformations or collections of vectors. | $A, B$ |
| **Dot Product** | An operation between two vectors that returns a scalar, measuring their alignment. | $\mathbf{a} \cdot \mathbf{b}$ |
| **Matrix Multiplication** | An operation where the number of columns in the first matrix must match the rows in the second. | $AB$ |
| **Linear Combination** | The sum of scaled vectors: $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_n\mathbf{v}_n$. | $\sum_{i=1}^{n} c_i \mathbf{v}_i$ |
| **Span** | The set of all possible linear combinations of a given set of vectors. | $\text{span}(\mathcal{B})$ |
| **Linearly Independent** | A set where no vector can be formed by a linear combination of the others. | $\text{LI}(\mathcal{B})$ |
| **Linearly Dependent** | A set where at least one vector is a linear combination of the others. | $\text{LD}(\mathcal{B})$ |
| **Basis** | A set that is both linearly independent and spans the entire space. | $\text{span}(\mathcal{B})=V \land \text{LI}(\mathcal{B})$ |
| **Zero Matrix** | A matrix consisting entirely of zeros. | $\mathbf{0}$ |
| **Matrix Transpose** | Flipping a matrix over its diagonal, switching rows and columns. | $A^\top$ |
| **Identity Matrix** | A square matrix with 1s on the main diagonal and 0s elsewhere. | $$ I_n $$ |
| **Determinant** | In short, the determinant represents the magnitude of the volume change a matrix can produce; it can provide important information about a square matrix, such as whether it is invertible. | det(A) |
| **Matrix Inversion** | The inverse matrix $A^{-1}$ is a matrix that undoes the effect of the linear transformation produced by $A$, returning the space to its original state. This is possible only if the transformation is "non-singular"—meaning its determinant is non-zero ($\det(A) \neq 0$). | $A^{-1}$ |
### 🔢 Formulas
$$c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$$

**Word-by-word explanation of the formula:**
* 📍 $c_{ij}$: means taking the row of the first matrix and the column of the second matrix.
* 🔄 $\sum_{k=1}^{n}$: represents summation that proceeds from the 1st term to the nth term.
* ✖️ $a_{ik} \cdot b_{kj}$: Multiply the $k$-th element of the column by the $k$-th element of the row.

### 🎯 Dot Product Formula
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n$$

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
