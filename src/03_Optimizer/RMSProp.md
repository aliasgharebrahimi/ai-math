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