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

<hr>

### Momentum
Well, momentum was introduced to address the problem with SGD; it adds a fraction of the previous update (velocity), scaled by a coefficient, to the current gradient.

Momentum utilizes the term $/gammo$ to incorporate a summary of previous gradients. Essentially, it uses this condensed representation of previous gradients and combines it with the current gradient.

Why do we use momentum? 

* **✅ Increased speed:** ✅ Momentum accumulates gradients in consistent directions, leading to larger parameter updates and faster convergence.
* **✅ Resolving the parameter-stuck issue:** In some instances, The gradient might be 10, for example, and then change to -10.; in such cases, the model's weights remain virtually unchanged.

<hr>

# RMSProp
An optimizer designed to address the issue of a uniform learning rate for all weights—it does so by utilizing the squares of past gradients and the square of the current gradient.

**Formula:**

$v(t) = β·v(t-1) + (1-β)·(∇L)²$

**Explanation of the formula:**

| Symbol | Explanation |
| :--- | :--- |
| $v(t)$ | It is the amount of parameter change derived from the squares of previous gradients and the square of the instantaneous gradient. |
| β·v(t-1) | This refers to the extent to which previous gradients are utilized—specifically, by multiplying them by a coefficient (this is known as momentum). |
| $(1-β)·(∇L)²$ | Here, it determines the remaining coefficient, multiplies it by the squared gradient to obtain the instantaneous gradient value, and adds this to the previous gradient. |