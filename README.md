# 🚀 ai-math
> Learning the mathematics required for deep learning, implementing it in code, and explaining the application of each concept within deep learning.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

<hr>

## ✨ Key Features
- **Mathematical Foundations:** Detailed conceptual definitions paired with step-by-step solved examples.
- **PyTorch Implementation:** All mathematical concepts are implemented using the PyTorch framework for practical learning.
- **Real-world Applications:** Explanations of how these topics apply to Deep Learning and Computer Vision.
- **Ready-to-run Code:** Streamlined implementation for quick execution and testing.

<hr>

## ⚙️ Prerequisites
### System Requirements
- **Operating System:** Windows, macOS, or Linux
- **Python Version:** `^3.9`
- **torch Version:** `^2.0`
- **Matplotlib Version:** `^3.7`

### Environment Setup (Recommended)

We recommend using `Conda` to manage dependencies:

```bash
# Create a new environment
conda create -n ai-math python=3.9
conda activate ai-math
```

<hr>

## 🚀 Installation & Setup
Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
First, clone this repository to your local system:

```bash
git clone https://github.com/alisgharebrahimi/ai-math.git
cd ai-math

# Install dependencies
pip install -r requirements.txt
```

<hr>

## 🚀 Usage
After completing the preparation steps, carry out these tasks:

Code has been written for each topic, structured using classes and methods. The main version of the code executes all the methods; however, you can choose not to run the main function that executes them all and instead run only the specific method you require.
To execute the method, simply call and use the desired function from the class object.

**Sample of the performance:**

```python
BasicCodes = BasicCodes(device=device, dtype=torch.float32, requires_grad=False)

inv = BasicCodes.matrix_inversion()
```

**Output code:**

<pre>
[20] Matrix Inversion A: tensor([[ 0.6000, -0.7000],
                                 [-0.2000,  0.4000]])
</pre>

<hr>

## 📁 Project Structure
<img src="./Project%20photos/Project%20Structure.jpg" alt="Project Structure">

```
ai-math/
├── 01_Linear Algebra/
│   ├── linear_algebra.py
│   └── linear_algebra.md
│   
├── 02_Differential/
│   ├── differential.py
│   └── differential.md
│   
├── 03_Optimizer/
│   ├── optimizer.py
│   └── optimizer.md
│        
├── README.md
├── LICENSE
└── requirements.txt
```

- **Linear Algebra:** Linear algebra concepts and related code
- **Differential:** Concepts of derivatives and differentials and their codes
- **README.md:** Project README.md
- **LICENSE:** Project License Text

<hr>

## 🤝 Contributing
Pull requests are welcome...

<hr>

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details