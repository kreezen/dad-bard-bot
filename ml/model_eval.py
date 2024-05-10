import torch
import torch.nn as nn
from torchvision import transforms
from datasetloader import GameFrameDataset

from model import SimpleCNN
from torch.utils.data import DataLoader
import torch.nn.functional as F


class ModelEvaluator:
    def __init__(self, model_path):
        """
        Initialize the ModelEvaluator.

        Args:
            model_path (str): The path to the pre-trained model checkpoint.
            num_classes (int): The number of classes in the classification problem.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def load_model(self, model_path):
        """
        Load a pre-trained model from a checkpoint file.

        Args:
            model_path (str): The path to the pre-trained model checkpoint.
            num_classes (int): The number of classes in the classification problem.

        Returns:
            torch.nn.Module: The loaded pre-trained model.
        """

        loaded_model = torch.load(model_path)
        return loaded_model

    def evaluate(self, dataloader):
        """
        Evaluate the loaded model on a given dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.

        Returns:
            float: The accuracy of the model on the evaluation dataset.
        """
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient calculation during evaluation
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                # Apply softmax to convert logits to probabilities
                probabilities = F.softmax(outputs, dim=1)

                # Get the class with the highest probability
                # predicted_class = torch.argmax(probabilities, dim=1)

                print(probabilities)

                # Get the confidence (probability) associated with the predicted class
                # confidence = probabilities[0][predicted_class].item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = correct_predictions / total_samples
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy


if __name__ == "__main__":
    model_evaluator = ModelEvaluator("./trained_models/bard.pth")
    # Define data transformations
    transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to the desired input size
                                    transforms.ToTensor()])  # Convert images to PyTorch tensors

    class_map = {'deadzone': 0, 'inner': 1, 'outer': 2}
    # Create dataset instances
    val_dataset = GameFrameDataset(
        root_dir='./dataset/validation', label_file='./dataset/validation/labels.txt', class_mapping=class_map, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    model_evaluator.evaluate(val_dataloader)
