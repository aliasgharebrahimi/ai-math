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
| $∇J(θ)$  | Loss gradient with respect to the parameter |
| $η$      | Learning rate |
| $θ ← θ$  |  Meaning the update of model parameters. |

**How does it work in mini-batch gradient descent?**

This mini-batch optimizer encompasses the entire set of samples.

# SGD

It bears a strong resemblance to GD, with the difference that SGD uses a single sample to update the parameters.

**Formula:**

The same GD formula

**Explanation of the formula:**

The same GD

**How does it work in mini-batch gradient descent?**

The main difference lies precisely here: SGD uses only a single sample.

**Important note:** In deep learning, the "mini-batch" is equivalent to the number of samples in the batch; this means, for instance, that SGD no longer operates on a single sample but instead processes the entire batch of samples.

# Momentum
Momentum was introduced to address a limitation of the Gradient Descent (GD) optimizer. The core idea is to align the current update with previous ones by accumulating a summary of past updates, thereby preventing oscillation.

**Formula:**

**First, the magnitude of the momentum:**

$$v_t = \beta v_{t-1} + \nabla J(\theta_t)$$

**And then update the parameters with momentum:**

$$\theta_{t+1} = \theta_t - \eta v_t$$

**Explanation of the formula:**

| Symbol   | Meaning |
|:---------| :--- |
| $v_t$ |  The variable representing the obtained momentum value |
| $\beta v_{t-1}$ | The magnitude of the momenta from previous stages (though not directly, since the $\beta$ coefficient utilizes only a portion of the information). |
| $\nabla J(\theta_t)$ | The magnitude of the loss gradient with respect to the parameters |

$\theta_{t+1} = \theta_t - \eta v_t$: The weight update formula is the same as the one used in the previous stages.

**Why do we use momentum?**

* **Using previous directions to reduce oscillation**
* **Optimizing routes**
* **Usually speeding up updates**