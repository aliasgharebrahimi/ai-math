import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
version = torch.__version__

class Differential:

    def partial_derivative(self, x, y):

        z = x**2 - y*2
        z.backward()

        return x.grad, y.grad

x = torch.tensor(9.0, requires_grad=True, device=device, dtype=torch.float32)
y = torch.tensor(6.0, requires_grad=True, device=device, dtype=torch.float32)

diff = Differential()

def main():

    p1, p2 = diff.partial_derivative(x, y)
    print(f"Partial derivative x: {p1}, Partial derivative y: {p2}")

if __name__ == ("__main__"):

    main()
