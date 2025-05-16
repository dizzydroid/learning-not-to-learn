import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Example: 2-layer CNN for MNIST
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*12*12, 128)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_features=128, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.fc(x)

class BiasPredictor(nn.Module):
    def __init__(self, in_features=128, num_bias=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_bias)
    def forward(self, x):
        return self.fc(x)
