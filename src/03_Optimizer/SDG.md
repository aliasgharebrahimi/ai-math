# SGD

It bears a strong resemblance to GD, with the difference that SGD uses a single sample to update the parameters.

**Formula:**

The same GD formula

**Explanation of the formula:**

The same GD

**How does it work in mini-batch gradient descent?**

The main difference lies precisely here: SGD uses only a single sample.

**Important note:** In deep learning, the "mini-batch" is equivalent to the number of samples in the batch; this means, for instance, that SGD no longer operates on a single sample but instead processes the entire batch of samples.