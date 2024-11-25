import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fifth Convolutional Block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)  # Increase dimensionality for a deeper layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)  # Output layer for two classes

    def forward(self, x):
        # Convolutional Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Convolutional Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Convolutional Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Convolutional Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # Convolutional Block 5
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Regularization
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Regularization
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer (no activation for logits)
        
        return x
