# 📐 Linear Algebra

---

##🔍 1. What is linear algebra?

> Linear algebra is the study of vectors an matrices, providing the mathematical framework and language essential for artificial intelligence.

---

## 🤖 2. So, where exactly is it used in artificial intelligenc?

As previously mentioned, the language of artificial intelligence is linear algebra (involving matrices, vectors, etc.), and it is utilized throughout the field. For instance:

* 🖼️ **Image Processing:** An image is fed int a model as a matrix.
* 🧠 **CNN Layers:** The kernel—which is itself  matrix—slides over the image.
* 🗺️ **Feature Maps:** The layer's outpu—a feature map—is also a matrix.
* 📉 **Jacobian Matrix:** Used for derivative (a concept we will also cover), which is also a matrix.

*Thus, it is evident that linear algebra is fundamental to every aspect of artificial intelligence.*

---

## ⚙️ 3. So, what exactly are matrix multiplication, matrix addition, and the dot product?

* **Matrix Multiplication:** The number of columns in the first matrix must match the number of rows in the second matrix. You multiply the rows of the first matrix by the columns of the second matrix—multiplying the first row by all the columns of the second matrix, then the second row by those columns, and so on.
* **Matrix Addition:** Matrix addition is very simple: you just add the corresponding elements from the two matrices.
* **Dot Product:** The dot product works similarly to addition, except that you multiply the corresponding elements instead of adding them.

---

## 📚 Linea Algebra Concepts

| Concept | Description |
| :--- | :--- |
| **Vectors** | A vector is an ordered list of numbers representing direction and magnitude in a multidimensional space. |
| **Matrices** | A collection of vectors arranged side-by-side in rows and columns is called a matrix. |
| **Matrix Multiplication** | Matrix multiplication differs from standard multiplication; the dimensions of the two matrices must align (specifically, the number of columns in the first matrix must match the number of rows in the second). The operation proceeds as follows: an element from the first matrix is ​​multiplied by the corresponding element in the second matrix's column, and so on. |
| **Matrix transpose** | In other words, we convert rows into columns and columns into rows; its notation is Aᵀ.|

### 🔢 The Multiplicatio Formula
$$c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$$

**Word-by-word explanation of the formula:**
* 📍 $c_{ij}$: means taking the row of the first matri and the column of the second matrix.
* 🔄 $\sum_{k=1}^{n}$: represents  summation that proceeds from the 1st term to the nth term.
* ✖️ $a_{ik} \cdot b_{kj}$: Multiply the $k$-th element of the column by the $k$-th element of the row.

### 🎯 Dot Product Formula
$$\mathbf{a} \cdot \mathbf{b} = sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n$$

---

## 💡 Importan Notes

> [!NOTE]
> **In PyTorch**, we refer to these vectors, matrices, and so on as **"tensors"**—a vector is a 1D tensor, a matrix is a 2D tensor, and so forth (3D tensor, etc.).

> [!TIP]
> **In CNN layers:**
> * ⚪ **Black-and-white images:** Represented as a matrix or 2D tensor.
> * 🌈 **RGB Color) images:** Represented as a 3D tensor (three matrices).

---

### � Alright, now that we understand what linear algebra is and where it’s used, let’s dive into the code and see it in action.