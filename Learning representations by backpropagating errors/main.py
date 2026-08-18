from utils.activations import sigmoid, d_sigmoid, relu, d_relu, tanh, d_tanh


def main():
    x = 0.5
    y = sigmoid(x)
    dy = d_sigmoid(x)
    print(f"y = {y}, dy = {dy}")
    y = relu(x)
    dy = d_relu(x)
    print(f"y = {y}, dy = {dy}")
    y = tanh(x)
    dy = d_tanh(x)
    print(f"y = {y}, dy = {dy}")
    print("Hello from learning-representations-by-backpropagating-errors!")


if __name__ == "__main__":
    main()
