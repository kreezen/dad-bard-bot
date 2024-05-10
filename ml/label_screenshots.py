import os
from PIL import Image
import torch
import torchvision.transforms as transforms

from model_eval import ModelEvaluator


class ScreenshotLabeler:
    def __init__(self, model, path_to_screenshots, filename="screens_labeled.txt") -> None:
        self.filename = filename
        self.model = model
        self.path_to_screenshots = path_to_screenshots
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def label_screenshots(self):
        if not os.path.exists(self.path_to_screenshots):
            print(f"Directory '{self.path_to_screenshots}' does not exist.")
            return
        screenshots = os.listdir(self.path_to_screenshots)
        print(screenshots)

        for screenshot in screenshots:
            pathname = screenshot
            screenshot = Image.open(self.path_to_screenshots + screenshot)
            screenshot = self.transform(screenshot)
            screenshot = torch.Tensor(screenshot.unsqueeze(0))
            screenshot = screenshot.to("cuda")
            predictions = self.model(screenshot)
            print(predictions)
            probabilities = torch.sigmoid(predictions)
            # print(probabilities)
            no_trigger, inner, outer = probabilities[0]
            no_trigger = no_trigger.item()
            inner = inner.item()
            outer = outer.item()

            label_text = ""
            if no_trigger > inner and no_trigger > outer:
                label_text = "deadzone"
            elif inner > no_trigger and inner > outer:
                label_text = "inner"
            else:
                label_text = "outer"

            with open(self.filename, 'a') as datei:
                datei.write(f"{pathname}, Label: {label_text}\n")
        print("done")


if __name__ == "__main__":
    model = ModelEvaluator("bard_97_8.pth")
    screen_labeler = ScreenshotLabeler(model.model, "./screens/")
    screen_labeler.label_screenshots()
