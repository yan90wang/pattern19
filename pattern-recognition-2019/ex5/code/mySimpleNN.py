import torch


class mySimpleNN(torch.nn.Module):
    '''
    Define the Neural network architecture
    '''

    def __init__(self, input_shape: tuple):
        super(mySimpleNN, self).__init__()
    # TODO: Define a simple neural network
        self.model = torch.nn.Sequential(
          torch.nn.Linear(input_shape, 10),
          torch.nn.ReLU(),
          torch.nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # TODO: Define the network forward propagation from x -> y_hat
        y_hat = self.model(x)
        return y_hat

