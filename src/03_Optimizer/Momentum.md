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