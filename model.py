import torch
import torch.nn as nn
import torch.nn.functional as F

# Example: Class weights in the loss function
class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in os.listdir(train_dir)]
weights = [sum(class_counts) / c for c in class_counts]  # Inverse frequency weighting
class_weights = torch.FloatTensor(weights).to(device)  # Move to GPU if needed

# CrossEntropyLoss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First Convolutional Layer: Input channels = 3 (RGB), Output channels = 32, Kernel Size = 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Output size: (32, H, W)
        # Second Convolutional Layer: Input channels = 32, Output channels = 64, Kernel Size = 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output size: (64, H/2, W/2)
        # Max Pooling Layer: Kernel Size = 2x2, Stride = 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully Connected Layer (input = 64 * 56 * 56, output = 512)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        # Second Fully Connected Layer (input = 512, output = 2)
        self.fc2 = nn.Linear(512, 2)  # Output 2 for two classes (e.g., Hemorrhagic, Ischemic)

    def forward(self, x):
        # Apply Conv1 + ReLU + MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        # Apply Conv2 + ReLU + MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 56 * 56)  # Flatten for fc layers
        # Apply Fully Connected Layer 1 + ReLU
        x = F.relu(self.fc1(x))
        # Apply Fully Connected Layer 2 (output layer)
        x = self.fc2(x)
        return x
