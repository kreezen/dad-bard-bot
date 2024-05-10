import os
from PIL import Image
from torch.utils.data import Dataset


class LabelMappingError(Exception):
    def __init__(self, label):
        self.label = label
        super().__init__(f"Label '{label}' not found in class_mapping")


class GameFrameDataset(Dataset):
    def __init__(self, root_dir, label_file, class_mapping=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_mapping = class_mapping if class_mapping else {}
        self.labels = self.load_labels(label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img_name, label = self.labels[idx]
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            return None, None

    def load_labels(self, label_file):
        labels = []
        try:
            with open(label_file, "r") as file:
                for line in file:
                    img_name, label = line.strip().split()
                    if label not in self.class_mapping:
                        raise LabelMappingError(label)
                    else:
                        label_id = self.class_mapping[label]
                    labels.append((img_name, label_id))
        except LabelMappingError as e:
            print(f"LabelMappingError: {str(e)}")
        except Exception as e:
            print(f"Error loading labels: {str(e)}")
        return labels
