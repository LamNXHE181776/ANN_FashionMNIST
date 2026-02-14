from torch import nn
from torch.nn import functional as F
import torch

class ANN(nn.Module):
    def __init__(self, shape, num_classes):
        super(ANN, self).__init__()

        with torch.no_grad():
            dummy_input = torch.zeros(shape)
            flattened_size = dummy_input.view(1, -1).size(1) #ANN input size is the flattened image size

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flattened_size, 2**8),
            nn.BatchNorm1d(2**8),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2**8, 2**7),
            nn.BatchNorm1d(2**7),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2**7, 2**6),
            nn.BatchNorm1d(2**6),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2**6, num_classes)
            )
    def forward(self, x):
        x = self.classifier(x)
        return x

