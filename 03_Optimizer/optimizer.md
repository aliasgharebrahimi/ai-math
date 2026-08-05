# Optimizer

---

> An optimizer is a deep learning algorithm used to update model parameters.

### family of optimizers:
```
(the progenitor)
├── SGD (Stochastic Gradient Descent)
├── Momentum
├── NAG (Nesterov)
├── AdaGrad
├── RMSprop
└── Adam (the most popular)
    └── AdamW
```

These all belong to the family of optimizers. Gradient Descent is the primary, foundational optimizer; the others utilize its formula but enhance accuracy by incorporating additional elements.

### Learning rate

In our optimizer formula, we utilize the gradient; however, because the gradient is large, we employ only a fraction of it—a value known as the learning rate (often denoted as LR).



