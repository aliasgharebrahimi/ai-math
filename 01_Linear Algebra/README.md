# Linear Algebra

- 1. ## What is linear algebra?
  
   - Linear algebra is the study of vectors and matrices, providing the mathematical framework and language essential for artificial intelligence.”

- 2. ## So, where exactly is it used in artificial intelligence?

  - As previously mentioned, the language of artificial intelligence is linear algebra (involving matrices, vectors, etc.), and it is utilized throughout the field. For instance, in image processing, an image is fed into a model as a matrix; within a CNN layer, the kernel—which is itself a matrix—slides over the image; the layer's output—a feature map—is also a matrix; and even the Jacobian matrix, used for derivatives (a concept we will also cover), is a matrix. Thus, it is evident that linear algebra is fundamental to every aspect of artificial intelligence.

- 3. ## So, what exactly are matrix multiplication, matrix addition, and the dot product?

  - First, regarding matrix multiplication: the number of columns in the first matrix must match the number of rows in the second matrix. You multiply the rows of the first matrix by the columns of the second matrix—multiplying the first row by all the columns of the second matrix, then the second row by those columns, and so on. Matrix addition is very simple: you just add the corresponding elements from the two matrices. The dot product works similarly to addition, except that you multiply the corresponding elements instead of adding them.

## Linear Algebra Concepts



- Dot product formula : $$ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n $$

## Note: In PyTorch, we refer to these vectors, matrices, and so on as "tensors"—a vector is a 1D tensor, a matrix is ​​a 2D tensor, and so forth (3D tensor, etc.).

## In CNN layers, color images are fed into the model as three matrices, while black-and-white images are fed as a single matrix.

- In computer vision, black-and-white images are represented as a matrix or 2D tensor, while RGB images are represented as a 3D tensor.

### Alright, now that we understand what linear algebra is and where it’s used, let’s dive into the code and see it in action.