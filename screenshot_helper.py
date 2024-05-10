from datetime import datetime
from enum import Enum, auto
from time import time
import cv2
import numpy as np
import pyautogui
import keyboard
import logging

# Configure logging
logging.basicConfig(filename='screenshot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


class State(Enum):
    INITIAL = auto()
    RUNNING = auto()
    STOPPED = auto()


class ScreenshotHelper:
    def __init__(self, capture_area=None, savepath=None, screenshotkey="j", shutdownkey="k"):
        keyboard.add_hotkey(shutdownkey, self.stop)
        self.screenshotkey = screenshotkey
        self.shutdownkey = shutdownkey
        self.savepath = savepath
        self.current_state = State.INITIAL
        self.capture_area = capture_area

    def start(self):
        self.current_state = State.RUNNING
        logging.info("Screenshot bot started.")
        while self.current_state == State.RUNNING:
            try:
                if keyboard.is_pressed(self.screenshotkey):
                    self.capture_screenshot()
            except Exception as e:
                logging.error(f"Error: {str(e)}")
                continue

    def stop(self):
        self.current_state = State.STOPPED
        logging.info("Screenshot bot stopped.")

    def capture_screenshot(self):
        try:
            screenshot = self.capture_screen_with_opencv()
            # Format the current UTC time as a string
            timestamp = time.time()
            save_path = f"{self.savepath}screenshot_{timestamp}.png"
            cv2.imwrite(save_path, screenshot)
            logging.info(f"Screenshot saved as {save_path}")

        except Exception as e:
            logging.error(f"Error capturing screenshot: {str(e)}")

    def capture_screen_with_opencv(self):
        try:
            if self.capture_area:
                x, y, width, height = self.capture_area
                screenshot = np.array(pyautogui.screenshot(
                    region=(x, y, width, height)))
            else:
                screenshot = np.array(pyautogui.screenshot())
            return cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Error capturing screen with OpenCV: {str(e)}")


if __name__ == "__main__":
    # Define the capture area (x, y, width, height)
    # capture_area = (1022, 1145, 519, 29)
    # Mouse Position: x=1524, y=1037 , 392, 23
    # Mouse Position: x=1916, y=1060
    capture_area = (1524, 1037, 392, 23)

    screenshot_helper = ScreenshotHelper(
        capture_area, savepath="screenshot_not_labeled/")
    screenshot_helper.start()

    try:
        screenshot_helper.start()
    except KeyboardInterrupt:
        screenshot_helper.stop()
    except Exception as e:
        logging.error(f"Unhandled error: {str(e)}")
