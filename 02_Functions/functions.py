# Import
import torch
import matplotlib.pyplot as plt 

# Defining a device to make the code more portable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Function example in Python
def f(x):
    return 2 * x
print(f(6.0)) # 12.0

# Example of a linear function in a CNN layer (at the corners): multiplying weights by the pixel value 
# In this example, we assume that this tensor is the very kernel that provided us with the weights.
tensor_Weights = torch.tensor([[0.4, 6.3, -9.0],
                              [-4.0, -8.0, 11.0],
                              [7.9, 1.12, -9.0]], dtype=torch.float32, device=device)

# In this example, too, we assume we are dealing with pixels—having converted the image into a tensor in the `transforms` section.
# which are numbers between 0 and 255 
# Note: We assume this image is black and white.
tensor_Pixel = torch.tensor([[0.4,  6.3,    9.0],
                             [1.0, -8.0,  19.0],
                             [12.9, 89.12, 9.0]], dtype=torch.float32, device=device)

# In CNN layers, our kernel slides the image and adds weights wherever it stops, and then it pixels a function *.
# Let us assume it is a mathematical function that executes in the layer.
# Mathematical function formula: y = ∑ᵢ₌₁ⁿ (xᵢ ⋅ wᵢ) 
# Finally, the bias is added to the result of the kernel multiplication and sliding operation.
# This function is executed thousands of times within the layer.
# Note: For efficiency, in practice, we use torch.matmul or optimized convolution operations.
# or aggregate functions
def Dot_Product(w, px):
    s = w * px
    g = torch.sum(s)

    return g

# ...so that after performing this operation on the entire image, the feature map is generated.
print(Dot_Product(tensor_Weights, tensor_Pixel))

# Nonlinear function

# The first method involves writing a custom function
#  based on a mathematical formula—for example, the sigmoid activation function: σ(x) = 1 / (1 + e⁻ˣ), where *e* (Euler's number) is approximately 2.718.
# The second method involves using PyTorch; while the functions involved do have underlying formulas, PyTorch handles the calculations itself.


x = torch.linspace(-10, 10, 100, device=device)

tensor_relu = torch.relu(x)

x_np = x.cpu().numpy()
y_np = tensor_relu.cpu().numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_np, y_np, label='ReLU Function', color='blue', linewidth=2)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3) # خط افقی روی صفر
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3) # خط عمودی روی صفر

plt.title('ReLU Activation Function', fontsize=14)
plt.xlabel('Input (x)', fontsize=12)
plt.ylabel('Output (ReLU(x))', fontsize=12)

plt.grid(True, which='both', linestyle='--', alpha=0.5) 
plt.legend() 

plt.show()