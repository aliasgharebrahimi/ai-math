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

| Concept | Description | Mathematical symbol |
| :--- | :--- | :--- |
| **Scalar** | A scalar is simply a number; in a space, it represents a point. | A, B,... |
| **Vectors** | A vector is an ordered list of numbers representing direction and magnitude in a multidimensional space which appears as a line in space. | A, B,... |
| **Matrices** | A collection of vectors arranged side-by-side in rows and columns is called a matrix And in space, the matrix takes the form of a two-dimensional plane, like a sheet. | A, B,... | 
| **Dot Product** | The dot product is used to measure the degree of similarity between two vectors, scalars, matrices, or other entities; for instance, with two vectors, the output value will be high if the vectors are similar, and low if they are not.| A.B |
| **Matrix Multiplication** | Note that matrix multiplication differs from ordinary multiplication; here, we use row-by-column multiplication Matrix multiplication differs from standard multiplication; the dimensions of the two matrices must align (specifically, the number of columns in the first matrix must match the number of rows in the second). The operation proceeds as follows: an element from the first matrix is multiplied by the corresponding element in the second matrix's column, and so on If the dimensions of our matrix are m by n and the second matrix is s by m, the dimensions of the output will be s by n. | A⋅B |
| **Linear combination** | It is very simple: it means multiplying several vectors by a scalar and then adding them together. | |
| **span** | The Span represents the set of all possible outcomes achievable through the linear combinations of a given set of vectors. It defines the total reachable space: for instance, if a linear combination only involves scaling a single vector, the span is restricted to a line (1D). However, if we introduce a second, linearly independent vector (allowing movement in a new direction), the span expands to cover a plane (2D) | |
| **Independent vector** | Well, an independent vector is defined this way: if the operation performed on the vector is independent—meaning it is not derived from a combination of other operations—then we call it an independent vector. Note that "operation" here refers to scalar multiplication and vector addition. | |
| **Zero matrix** | It refers to a matrix in which all the numbers are 0. | 0 |
| **Matrix transpose** | In other words, we convert rows into columns and columns into rows; its notation is Aᵀ.| Aᵀ
| **Identity matrix** | A square matrix with 1s on the diagonal and 0s elsewhere, denoted by I_n (where n represents the number of rows/columns); for example, I_3 refers to an identity matrix with 3 rows and 3 columns.| Iₙ
| **Matrix Inversion Operation** | In other words, if a matrix is multiplied by its inverse, the result is the identity matrix; furthermore, for a matrix to have an inverse, it must be square and its determinant must be non-zero. | 

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
