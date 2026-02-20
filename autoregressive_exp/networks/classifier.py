import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the image classifier model
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        
        # 1st block: 1 input channel (for grayscale MNIST) -> 32 output channels
        # Kernel size 3x3, Padding 1 ensures output size remains 28x28 before pooling
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 2nd block: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Dropout layer for regularization
        self.dropout1 = nn.Dropout(0.25)
        
        # 3rd block: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Final dropout before fully connected layers
        self.dropout2 = nn.Dropout(0.5)
        
        # After two MaxPool2d (kernel_size=2) operations, the spatial size 
        # is reduced from 28x28 -> 14x14 -> 7x7. 
        # The output of the last conv layer is 128 channels, so 128 * 7 * 7
        self.fc1 = nn.Linear(128 * 7 * 7, 512) 
        self.fc2 = nn.Linear(512, 10) # 10 output classes (digits 0-9)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) # Size becomes 14x14
        
        # Block 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) # Size becomes 7x7
        x = self.dropout1(x)
        
        # Block 3
        x = F.relu(self.conv3(x))
        
        # Flatten for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # The loss function (nn.CrossEntropyLoss) typically includes Softmax, 
        # so we return the raw logits.
        return x