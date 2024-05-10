import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, num_conv_layers=5, neurons_widght_height=7):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.neurons_widght_height = neurons_widght_height
        in_channels = 3
        out_channels = 16
        self.conv_layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

        for _ in range(num_conv_layers - 1):
            in_channels = out_channels
            out_channels *= 2
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # print(f"output: {out_channels * neurons_widght_height * neurons_widght_height}")
        self.fc1 = nn.Linear(
            out_channels * neurons_widght_height * neurons_widght_height, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = self.pool(torch.relu(conv_layer(x)))

        # print(f"shape: {self.fc1.in_features}")

        # Flatten
        x = x.view(-1, self.fc1.in_features)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x
