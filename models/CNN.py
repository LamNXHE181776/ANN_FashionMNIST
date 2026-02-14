from torch import nn
import torch
from torch.nn import functional as F
from torchinfo import summary

class CNN(nn.Module):
    def __init__(self, shape, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            nn.Conv2d(in_channels = 64, out_channels = 81, kernel_size=3, padding="same"),
            nn.BatchNorm2d(81),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            nn.Conv2d(in_channels = 81, out_channels = 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Dropout2d(0.3),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(shape)
            features_output = self.features(dummy_input)
            flattened_size = features_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flattened_size, 2**8),
            nn.BatchNorm1d(2**8),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2**8, num_classes)
            )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = CNN(shape=(1, 1, 28, 28), num_classes=10)
summary(model, input_size=(1,1,28,28))