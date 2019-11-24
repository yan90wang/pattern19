import torch


class MyLogRegNN(torch.nn.Module):

    def __init__(self):
        super(MyLogRegNN, self).__init__()
        # TODO: Define a logistic regression classifier as a neural network
        self.logReg = torch.nn.Linear(3072, 32)
    def forward(self, x):
        y_hat = torch.sigmoid(self.logReg(x.view(-1, x.shape[0])))
        return y_hat


class MyFullyConnectedNN(torch.nn.Module):
    def __init__(self):
        super(MyFullyConnectedNN, self).__init__()
        # TODO: Define a fully connected neural network
        self.model = torch.nn.Sequential(
          torch.nn.Linear(3072, 1024),
          torch.nn.ReLU(),
          torch.nn.Linear(1024, 32),
        )

    def forward(self, x):
        y_hat = self.model(x.view(-1, x.shape[0]))
        return y_hat


class MyCNN(torch.nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()
        self.model = torch.nn.Conv2d(3, 6, 3, stride=2)
        # TODO: Define a convolutional neural network

    def forward(self, x):
        y_hat = None
        return y_hat
