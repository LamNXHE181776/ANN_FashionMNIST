from torch import nn
from torch.nn import functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(2**8)
        self.batchnorm2 = nn.BatchNorm1d(2**7)
        self.batchnorm3 = nn.BatchNorm1d(2**6)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(input_size, 2**8)
        self.linear2 = nn.Linear(2**8, 2**7)
        self.linear3 = nn.Linear(2**7, 2**6)
        self.linear4 = nn.Linear(2**6, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.batchnorm3(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.linear4(x)
        return x