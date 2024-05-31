from datasetloader import GameFrameDataset
from model import SimpleCNN
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models


class Train():
    def __init__(self, model, criterion, optimizer, device, train_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        # torch.save(self.model, model_path)
        print("Model saved.")

    def train_model(self, num_epochs, threshold=0.97):

        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode

            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # Zero the gradient buffers
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            train_loss = running_loss / len(self.train_loader)
            train_accuracy = (correct_predictions / total_samples) * 100.0
            if train_accuracy >= threshold:
                self.save_model("bard.pth")
                break

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%")
        print("Training finished.")


def main():
    # Define data transformations
    transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to the desired input size
                                    transforms.ToTensor(),  # Convert images to PyTorch tensors
                                    ])

    # Map class names to integers
    class_map = {'deadzone': 0, 'inner': 1, 'outer': 2}
    # Create dataset instances
    train_dataset = GameFrameDataset(root_dir='dataset/training',
                                     label_file='./dataset/training/labels.txt',
                                     class_mapping=class_map,
                                     transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Define Configurations
    model_cnn = SimpleCNN(num_classes=3, num_conv_layers=5,
                          neurons_widght_height=7)

    # Cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with learning rate
    optimizer = torch.optim.Adam(model_cnn.parameters(), lr=0.001)

    print(f"cuda: {torch.cuda.is_available()}")
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the GPU if available
    model_cnn.to(device)

    # Define the number of epochs, ends anyways when the thresholds hits
    num_epochs = 200
    threshold = 0.98

    # Start training
    train = Train(model_cnn, criterion, optimizer, device, train_loader)
    train.train_model(num_epochs, threshold=threshold)
    train.save_model("trained_models/bard.pth")


if __name__ == "__main__":
    main()
