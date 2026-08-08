# optimizer
> In machine learning, there is an algorithm that reduces the error (loss) by adjusting the model's parameters.

<hr>

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

<hr>



<hr>

# SGD

It bears a strong resemblance to GD, with the difference that SGD uses a single sample to update the parameters.

**Formula:**

The same GD formula

**Explanation of the formula:**

The same GD

**How does it work in mini-batch gradient descent?**

The main difference lies precisely here: SGD uses only a single sample.

**Important note:** In deep learning, the "mini-batch" is equivalent to the number of samples in the batch; this means, for instance, that SGD no longer operates on a single sample but instead processes the entire batch of samples.

<hr>

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

<hr>

# RMSProp
The core idea is very clever: first, the update magnitude for parameters and gradients is calculated individually for each parameter; second, by incorporating the squares of past gradients, the update step is controlled and optimized.
An interesting point is that the learning rate varies for each parameter!
In this optimizer, a larger gradient results in a smaller update, leading to a more appropriate update.

Note: This optimizer only preserves the ratio of the previous values, but not the direction.

**Formula:**

First, calculate the squares of the previous gradients:

$$s_t = \beta s_{t-1} + (1-\beta)(\nabla J(\theta_t))^2$$

Parameter update :

$$\theta_{t+1}= \theta_t- \frac{\eta}{\sqrt{s_t}+\epsilon} \nabla J(\theta_t)$$

**The task of both formulas:**

$s_t = \beta s_{t-1} + (1-\beta)(\nabla J(\theta_t))^2$: Obtaining the square of previous gradients and the current gradient.

$\theta_{t+1}= \theta_t- \frac{\eta}{\sqrt{s_t}+\epsilon} \nabla J(\theta_t)$: Weight update

$\frac{\eta}{\sqrt{s_t}+\epsilon}$: This is the crux of the matter: a variable learning rate proportional to the gradient.

**Mini-batch optimizer:**

All samples

<hr>

# Adam
Well, here we are at the world's most famous and widely used optimizer. Why? Because it maintains direction *and* applies a different learning rate for each parameter—essentially combining Momentum and RMSProp.