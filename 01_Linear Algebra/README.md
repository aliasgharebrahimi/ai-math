# Linear Algebra

- 1. What is linear algebra?
  
  ## Linear algebra is the study of vectors and matrices, providing the mathematical framework and language essential for artificial intelligence.”

- 2. So, where exactly is it used in artificial intelligence?

  - As previously mentioned, the language of artificial intelligence is linear algebra (involving matrices, vectors, etc.), and it is utilized throughout the field. For instance, in image processing, an image is fed into a model as a matrix; within a CNN layer, the kernel—which is itself a matrix—slides over the image; the layer's output—a feature map—is also a matrix; and even the Jacobian matrix, used for derivatives (a concept we will also cover), is a matrix. Thus, it is evident that linear algebra is fundamental to every aspect of artificial intelligence.

## Note: In PyTorch, we refer to these vectors, matrices, and so on as "tensors"—a vector is a 1D tensor, a matrix is ​​a 2D tensor, and so forth (3D tensor, etc.).

- In computer vision, black-and-white images are represented as a matrix or 2D tensor, while RGB images are represented as a 3D tensor.

### Alright, now that we understand what linear algebra is and where it’s used, let’s dive into the code and see it in action.