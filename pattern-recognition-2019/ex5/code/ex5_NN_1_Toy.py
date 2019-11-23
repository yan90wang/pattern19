import sys
import torch


def toyNetwork() -> None:
    # TODO: Implement network as given in the exercise sheet.
    # Manual implementation functionality when computing loss, gradients and optimization
    # i.e. do not use torch.optim or any of the torch.nn functionality
    # Torch documentation: https://pytorch.org/docs/stable/index.html

    # TODO: Define weight variables using: torch.tensor([], requires_grad=True)
    # TODO: Define data: x, y using torch.tensor
    # TODO: Define learning rate

    # TODO: Train network until convergence
    # TODO: Define network forward pass connectivity
    # TODO: Get gradients of weights and manually update the network weights

    # Steps:
    # 1 - compute error
    # 2 - do backward propagation, use: error.backward() to do so
    # 3 - update weight variables according to gradient and learning rate
    # 4 - Zero weight gradients with w_.grad_zero_()

    w1 = torch.tensor([0.5], requires_grad=True)
    w3 = torch.tensor([0.3], requires_grad=True)
    w5 = torch.tensor([0.8], requires_grad=True)
    w2 = torch.tensor([0.3], requires_grad=True)
    w4 = torch.tensor([0.1], requires_grad=True)
    w6 = torch.tensor([0.3], requires_grad=True)
    w7 = torch.tensor([0.5], requires_grad=True)
    w8 = torch.tensor([0.9], requires_grad=True)
    w9 = torch.tensor([0.2], requires_grad=True)

    learning_rate = 0.2
    x = torch.tensor([1.0, 1.0])
    y_true = torch.tensor([1.0])
    old_loss = y_true

    for i in range(100):
        h1 = torch.sigmoid(x[0]*w1 + x[1]*w3 + w5)
        print("h1: " + str(h1.item()))
        h2 = torch.sigmoid(x[0]*w2 + x[1]*w4 + w6)
        print("h2: " + str(h2.item()))
        y = h1*w7 + h2*w8 + w9
        loss = 0.5 * (y_true - y).pow(2)
        print("loss: " + str(loss.item()))

        if abs(old_loss - loss) < 1e-5:
            break
        else:
            old_loss = loss

        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            print("new w1: " + str(w1.item()))
            w2 -= learning_rate * w2.grad
            print("new w2: " + str(w2.item()))
            w3 -= learning_rate * w3.grad
            print("new w3: " + str(w3.item()))
            w4 -= learning_rate * w4.grad
            print("new w4: " + str(w4.item()))
            w5 -= learning_rate * w5.grad
            print("new w5: " + str(w5.item()))
            w6 -= learning_rate * w6.grad
            print("new w6: " + str(w6.item()))
            w7 -= learning_rate * w7.grad
            print("new w7: " + str(w7.item()))
            w8 -= learning_rate * w8.grad
            print("new w8: " + str(w8.item()))
            w9 -= learning_rate * w9.grad
            print("new w9: " + str(w9.item()))

            w1.grad.zero_()
            w2.grad.zero_()
            w3.grad.zero_()
            w4.grad.zero_()
            w5.grad.zero_()
            w6.grad.zero_()
            w7.grad.zero_()
            w8.grad.zero_()
            w9.grad.zero_()

        print("##########-##########-##########" + "end iter: " + str(i))

if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network toy example!")
    toyNetwork()
    print("Done!")
