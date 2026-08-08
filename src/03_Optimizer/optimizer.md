# optimizer
> In machine learning, there is an algorithm that reduces the error (loss) by adjusting the model's parameters.

# family of optimizers
```
GD(the progenitor)
├── SGD (Stochastic Gradient Descent)
├── Momentum
├── NAG (Nesterov)
├── AdaGrad
├── RMSprop
└── Adam (the most popular)
    └── AdamW
```

# GD
This is one of the earliest and simplest optimizers; other optimizers, such as SGD, were built upon its algorithm.

**Formula:**

$θ ← θ - η ∇J(θ)$

**Explanation of the formula:**

| Symbol   | Meaning |
|:---------| :--- |
| $∇J(θ)$  | Average of parameter mini-batch gradients |
| $η$      | Learning rate |
| $θ ← θ$  |  Meaning the update of model parameters. |

**How does it work in mini-batch gradient descent?**

This mini-batch optimizer encompasses the entire set of samples.