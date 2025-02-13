import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN2D(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(SimpleCNN2D, self).__init__()
        
        # First Conv Layer with "SAME" padding and ReLU activation
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same')
        
        # First MaxPooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Second Conv Layer with ReLU activation
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='valid')
        
        # Second MaxPooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Third Conv Layer with ReLU activation
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='valid')
        
        # Second MaxPooling layer
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # # Third Conv Layer with ReLU activation
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='valid')
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, max_frames, input_features * num_coords)  # batch size of 1
            flattened_size = self._get_flattened_size(dummy_input)

        print("======> Size:", flattened_size)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_signs)
    
    def _get_flattened_size(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1).size(1)  # Flatten with batch dimension kept

    def forward(self, x):
        # Forward pass through Conv and Pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Maintain batch dimension
        
        # Fully connected layers with Dropout and Softmax
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x  # Logits returned for CrossEntropyLoss
    

class ComplexCNN2D(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(ComplexCNN2D, self).__init__()
        
        # First Convolution Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, input_features * num_coords), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1))
        
        # Second Convolution Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, input_features * num_coords), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1))
        
        # Third Convolution Block with Residual Connection
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, input_features * num_coords), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1))
        
        self.residual = nn.Conv2d(64, 128, kernel_size=1, stride=1)  # To match dimensions for residual
        
        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, input_features * num_coords), padding='valid')
        # self.bn4 = nn.BatchNorm2d(256)
        # self.pool4 = nn.MaxPool2d(kernel_size=(3, 1))
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, max_frames, input_features * num_coords)
            flattened_size = self._get_flattened_size(dummy_input)

        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, num_signs)

    def _get_flattened_size(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block with residual connection
        residual = self.residual(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))) + residual)
        
        x = (F.relu((self.conv4(x))))
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block with residual connection
        residual = self.residual(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))) + residual)
        
        # Fourth block
        x = (F.relu((self.conv4(x))))
        
        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
if __name__ == '__main__':
    print("Running cnn_2d.py")
    model = SimpleCNN2D(60, 21, 2, 563)
    print(model)