# 📐 Linear Algebra for AI

---

## 🔍 1. What is Linear Algebra?

> Linear algebra is the study of vectors and matrices, providing the mathematical framework and language essential for artificial intelligence.

---

## 🤖 2. Where is it used in Artificial Intelligence?

As previously mentioned, the language of artificial intelligence is linear algebra (involving matrices, vectors, etc.), and it is utilized throughout the field. For instance:

* 🖼️ **Image Processing:** An image is fed **into** a model as a matrix.
* 🧠 **CNN Layers:** The kernel—which is itself a matrix—slides over the image.
* 🗺️ **Feature Maps:** The layer's **output** (a feature map) is also a matrix.
* 📉 **Jacobian Matrix:** Used for derivatives (a concept we will also cover), which is also a matrix.

*Thus, it is evident that linear algebra is fundamental to every aspect of artificial intelligence.*

---

## 📚 Linear Algebra Concepts

| Concept | Description | Mathematical Symbol |
| :--- | :--- | :--- |
| **Scalar** | A scalar is simply a number; in a space, it represents a point. | $a, b, ...$ |
| **Vectors** | A vector is an ordered list of numbers representing direction and magnitude in a multidimensional space. It appears as a line in space. | $\mathbf{a}, \mathbf{b}, ...$ |
| **Matrices** | A collection of vectors arranged side-by-side in rows and columns is called a matrix. In space, the matrix takes the form of a two-dimensional plane. | $\mathbf{A}, \mathbf{B}, ...$ | 
| **Dot Product** | The dot product measures the degree of similarity between two vectors. If vectors are similar, the output is high; if not, it is low. | $\mathbf{a} \cdot \mathbf{b}$ |
| **Matrix Multiplication** | Row-by-column multiplication. The number of columns in the first matrix must match the number of rows in the second. If dimensions are $m \times n$ and $n \times p$, the result is $m \times p$. | $\mathbf{A} \cdot \mathbf{B}$ |
| **Zero Matrix** | A matrix in which all elements are 0. | $\mathbf{0}$ |
| **Matrix Transpose** | Converting rows into columns and columns into rows. | $\mathbf{A}^T$ |
| **Identity Matrix** | A square matrix with 1s on the diagonal and 0s elsewhere. Denoted by $I_n$ (where $n$ is the number of rows/columns). | $\mathbf{I}_n$ |
| **Matrix Inversion** | If a matrix is multiplied by its inverse, the result is the identity matrix. The matrix must be square and have a non-zero determinant. | $\mathbf{A}^{-1}$ |

---

### 🔢 Formulas

#### 1. Matrix Multiplication Formula
$$c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$$

**Word-by-word explanation:**
* 📍 $c_{ij}$: The element in the $i$-th row and $j$-th column of the resulting matrix.
* 🔄 $\sum_{k=1}^{n}$: Summation from the 1st term to the $n$-th term.
* ✖️ $a_{ik} \cdot b_{kj}$: Multiply the $k$-th element of the row from the first matrix by the $k$-th element of the column from the second matrix.

#### 2. Dot Product Formula
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n$$

---

## 💡 Important Notes

> [!NOTE]
> **In PyTorch**, we refer to these vectors, matrices, and so on as **"tensors"**:
> * A scalar is a 0D tensor.
> * A vector is a 1D tensor.
> * A matrix is a 2D tensor.
> * And so on...

> [!TIP]
> **In CNN Layers:**
> * ⚪ **Black-and-white images:** Represented as a 2D tensor (matrix).
> * 🌈 **RGB Color images:** Represented as a 3D tensor (three matrices/channels).

---