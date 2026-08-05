# Optimizer

---

> An optimizer is a deep learning algorithm used to update model parameters.

### family of optimizers:
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

These all belong to the family of optimizers. Gradient Descent is the primary, foundational optimizer; the others utilize its formula but enhance accuracy by incorporating additional elements.

<hr>

### Learning rate
In our optimizer formula, we utilize the gradient; however, because the gradient is large, we employ only a fraction of it—a value known as the learning rate (often denoted as LR).

<hr>

### SGD, GD

SGD is also an optimizer, and its formula is as follows:

$θ = θ - η * ∇θ J(θ)$

Of course, the GD formula is also like this:

$θ = θ - η * ∇θ J(θ)$

The formulas for the two are exactly the same, but the difference lies in the gradients: SGD calculates the average gradient for a subset of the model's parameters, whereas GD calculates it for all of them.

Advantages and disadvantages of each:

SGD:

✅ Faster

❌ Fluctuating

GD:

✅ More precisely

❌ Slower