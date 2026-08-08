# GD
This is one of the earliest and simplest optimizers; other optimizers, such as SGD, were built upon its algorithm.

**Formula:**

$θ ← θ - η ∇J(θ)$

**Explanation of the formula:**

| Symbol   | Meaning                                          |
|:---------|:-------------------------------------------------|
| $∇J(θ)$  | Loss gradient with respect to the parameter and A combination of the current gradient and previous momenta. |
| $η$      | Learning rate                                    |
| $θ ← θ$  | Meaning the update of model parameters.          |

**How does it work in mini-batch gradient descent?**

This mini-batch optimizer encompasses the entire set of samples.