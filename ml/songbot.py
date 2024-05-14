from enum import Enum, auto
import os
import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab
import torch
import torch.nn as nn
from torchvision import transforms
import time


class Box(Enum):
    INTIAL = auto()
    OUTER = auto()
    INNER = auto()


class Clikk:
    def __init__(self, box: Box) -> None:
        self.allowed_mouse_clikk = box == Box.OUTER


class ScreenCaptureNN:
    def __init__(self, model_path, save_path=None, interval=1.0, screen_area=(0, 0, 800, 600)):
        self.save_path = save_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        # Define the specific screen area to capture (left, top, right, bottom)
        self.screen_area = screen_area
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.interval = interval
        self.mouse_clikk = Clikk(Box.INTIAL)

    def capture_screen(self):
        screen = ImageGrab.grab(bbox=self.screen_area)
        screen = screen.convert('RGB')
        trans = self.transform(screen).unsqueeze(0).to(self.device)
        return screen, trans

    def save_screenshot(self, screen):
        # Generate a unique filename based on the current timestamp
        timestamp = time.time()
        screenshot_filename = f"screenshot_{timestamp}.png"
        save_path = os.path.join(self.save_path, screenshot_filename)
        screenshot = np.array(screen)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, screenshot)
        print(f"Screenshot saved to {save_path}")

    def start_loop(self, threshold=0.8):
        print("Starting loop...")
        while True:
            self.evaluate_screen(threshold=threshold)
            time.sleep(self.interval)

    def evaluate_screen(self, threshold=0.8):
        screen, trans = self.capture_screen()

        with torch.no_grad():
            predictions = self.model(trans)
            probabilities = torch.sigmoid(predictions)
            deadzone, inner, outer = probabilities[0]
            # print(inner.item(), outer.item())

        if outer.item() >= threshold:
            self.mouse_clikk = Clikk(Box.OUTER)
            if self.save_path:
                self.save_screenshot(screen)

        if inner.item() >= threshold:
            # If the prediction meets the threshold, trigger a right mouse click
            # if self.mouse_clikk.allowed_mouse_clikk:
            pyautogui.mouseDown(button='right')
            self.mouse_clikk = Clikk(Box.INNER)
            pyautogui.mouseUp(button='right')
            # print("Right click triggered")

            # Capture and save a screenshot
            if self.save_path:
                self.save_screenshot(screen)

            time.sleep(0.1)

        else:
            # Otherwise, do nothing
            pass
            # print("No action taken")


if __name__ == "__main__":

    # Update with your PyTorch model path
    model_path = "./trained_models/bard_97_8.pth"

    # Set the interval (in seconds) between taking screenshots
    interval = 0.01

    # Screenshots will be disabled when None
    # good for more testing data
    # save_path = "bot_screenshots"
    save_path = None

    # use these to finetune the cropping area
    offset_height = 5
    offset_width = 0

    # take an ingame screenshot and Fullscreen it
    # get the position of the upper left corner and the lower right corner of the musicbox
    # use mousecursor.py to find the upper left corner and the lower right corner of the musicbox
    # replace the values with your position
    # this position works for 2k ingame resolution
    left_corner_position_of_musicbox = {"x": 1014, "y": 1143}
    right_corner_position_of_musicbox = {"x": 1542, "y": 1172}

    # define the cropping area
    x = left_corner_position_of_musicbox["x"]
    y = left_corner_position_of_musicbox["y"]
    width = right_corner_position_of_musicbox["x"] - \
        left_corner_position_of_musicbox["x"] + offset_width
    height = right_corner_position_of_musicbox["y"] - \
        left_corner_position_of_musicbox["y"] + offset_height

    # Calculate the left, top, right, and bottom coordinates for ImageGrab.grab
    left = x
    top = y
    right = x + width
    bottom = y + height
    bbox = (left, top, right, bottom)

    screen_capture_nn = ScreenCaptureNN(
        model_path, interval=interval, screen_area=bbox, save_path=save_path)

    # Model prediction threshold, needs to be sure at least 80% to trigger
    threshold = 0.8
    screen_capture_nn.start_loop(threshold)
