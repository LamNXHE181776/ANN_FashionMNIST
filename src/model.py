from torch import nn
from torch.nn import functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size, 2**9)
        self.linear2 = nn.Linear(2**9, 2**9)
        self.linear3 = nn.Linear(2**9, 2**8)
        self.linear4 = nn.Linear(2**8, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x