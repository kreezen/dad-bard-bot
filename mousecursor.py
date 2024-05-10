import pyautogui
import keyboard

# Variable to track whether the mouse position display is active
display_active = False


def display_mouse_position():
    global display_active
    display_active = True
    while display_active:
        x, y = pyautogui.position()
        print(f"Mouse Position: x={x}, y={y}")


def stop_display():
    global display_active
    display_active = False


if __name__ == '__main__':
    # Create a keyboard listener for the 's' key to stop the display
    keyboard.add_hotkey('s', stop_display)

    # Use this to get the mouse position from the upper and lower left corners of the box of an ingame screenshot
    display_mouse_position()
