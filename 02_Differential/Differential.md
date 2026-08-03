# 📐 Partial Derivative

---

### 💡 Conceptual Definition
The **partial derivative** represents the instantaneous rate of change of a multivariable function's output with respect to **one specific variable**, while all other variables are held **constant**.

---

### 🔢 Mathematical Definition
To compute the partial derivative with respect to a specific variable, we treat all other variables as fixed constants and perform the differentiation process solely on the variable of interest.

The operation is denoted as:
$$\frac{\partial f}{\partial x}$$

---

### 🔍 Symbol Analysis
In the expression $\frac{\partial f}{\partial x}$:

| Symbol | Meaning |
| :--- | :--- |
| $\partial$ | Denotes the **partial** derivative operation (distinguishing it from total derivatives). |
| $f$ | Represents the **multivariable function** under consideration. |
| $x$ | The **target variable** with respect to which the derivative is taken. |

---

### 🧠 Application in Deep Learning
Partial derivatives are the backbone of modern AI. They are fundamental to:

*   **Backpropagation:** The engine that allows neural networks to learn.
*   **Gradient Descent:** Calculating the **gradient** ($\nabla f$), which is a vector of partial derivatives used to update model weights.
*   **The Chain Rule:** Essential for propagating errors through complex, nested computational graphs.

---

# Gradient

### 💡 Conceptual Definition

The gradient is a **vector** containing the partial derivatives of all the **variables of a given function.**

### 🔢 Mathematical Definition

First, we calculate the **partial derivatives** of the function with respect to all its variables, and then we arrange them **together as a vector.**
which is denoted by $\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)$

## 🔍 Symbol Analysis
In the expression $\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)$:

| Symbol                            | Meaning            |
|:----------------------------------|:-------------------|
| $\nabla f$                        | Gradient symbol    |
| $\frac{\partial f}{\partial x_1}$ | Partial derivative |
| $\frac{\partial f}{\partial x_2}$ | Partial derivative |
| $, \dots,$ |   In other words, we can include any number of partial derivatives.                 |

### 🧠 Application in Deep Learning

This gradient constitutes a core component of deep learning; without it, the training process would become highly unstable.

*   **The Chain Rule** Using the gradient vector, the multiplication associated with the chain rule can be performed.
*   **Backpropagation:** We apply the chain rule using the gradient vector to enable backpropagation.

---

# Chain Rule

### 💡 Conceptual Definition

When differentiating a function like $f(x, y)$, we would simply calculate the derivative; however, in fields such as deep learning, functions are often composite—meaning one function uses the output of another as its input. If we want to find the derivative of a variable within the first function with respect to the output of the second function, we must use the chain rule.

### 🔢 Mathematical Definition

This is a composite function that also utilizes the outputs:

$g(x, y) = x + y$

$f(g(x, y)) = g(x, y)^2$

To find the partial derivative of the function $f$ with respect to $x$, we must first determine the partial derivative of $g$ with respect to $x$ and the partial derivative of $f$ with respect to $g$, and then multiply the two derivatives together:

$$ \frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x} $$