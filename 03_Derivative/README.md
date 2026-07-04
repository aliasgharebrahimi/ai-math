# Derivative

- 1. Well, first of all, what exactly is a derivative?
  
  ## A derivative represents the instantaneous rate of change of a function at a specific point.

- 2. What is a gradient?

  ## The gradient is a vector of partial derivatives. While a derivative describes the rate of change for a single variable, the gradient generalizes this concept to multi-variable functions, pointing in the direction of the steepest ascent in a multi-dimensional space.

- 3. Where does it fit into the landscape of artificial intelligence, and what is PyTorch's role?

  ## First, weights are initialized and a loss is calculated. During backpropagation, we compute the gradient of the loss with respect to the activation function, and then the gradient of the activation with respect to each weight. Using the Jacobian matrix and the Chain Rule, these local gradients are multiplied to propagate the error backward, ultimately updating each weight to minimize the loss.

- 4. Alright, let's use mathematics to explain this very wheel.
  
### Given:
- $w = [w_1, w_2]$ (Weights)
- $a(w) = \sum (w \cdot 2)$ (Linear Layer)
- $r(a) = \max(0, a)$ (Activation Function - ReLU)

### Goal: Calculate the gradient of $r$ with respect to $w$.

### Jacobian Matrix Concept:
The relationship between the activation and the weights can be represented as:
$$J = \begin{bmatrix} \frac{\partial a}{\partial w_1} & \frac{\partial a}{\partial w_2} \\ \frac{\partial r}{\partial a} & 0 \end{bmatrix}$$
*(Note: In practice, we use Vector-Jacobian Products for efficiency)*

### Chain Rule Application:
To find the gradient for each weight, we multiply the local derivatives:
- $\frac{\partial r}{\partial w_1} = \frac{\partial r}{\partial a} \cdot \frac{\partial a}{\partial w_1}$
- $\frac{\partial r}{\partial w_2} = \frac{\partial r}{\partial a} \cdot \frac{\partial a}{\partial w_2}$