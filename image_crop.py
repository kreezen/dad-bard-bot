import os
from PIL import Image


def crop_images_in_directory(directory_path, crop_box):
    """
    Crop all images in a directory using a specified cropping box.

    Args:
        directory_path (str): Path to the directory containing the images.
        crop_box (tuple): A tuple (left, upper, right, lower) specifying the cropping box.
                          For example, (100, 100, 400, 400) will crop a 300x300 pixel area.

    Returns:
        None
    """
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # List all files in the directory
    file_list = os.listdir(directory_path)

    # Filter out only image files (you can add more extensions as needed)
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
    image_files = [file for file in file_list if any(
        file.lower().endswith(ext) for ext in image_extensions)]

    if not image_files:
        print(f"No image files found in '{directory_path}'.")
        return

    # Create a subdirectory for the cropped images
    output_directory = os.path.join(directory_path, "cropped_images")
    os.makedirs(output_directory, exist_ok=True)

    # Loop through each image and crop it
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        try:
            # Open the image using Pillow
            img = Image.open(image_path)

            # Crop the image using the provided crop box
            cropped_img = img.crop(crop_box)

            # Save the cropped image to the output directory
            output_path = os.path.join(output_directory, image_file)
            cropped_img.save(output_path)

            print(f"Cropped '{image_file}' and saved as '{output_path}'.")
        except Exception as e:
            print(f"Error processing '{image_file}': {str(e)}")

# Example usage:
# Define your cropping box here
# crop_box = (69, 45, 590, 75)
# crop_images_in_directory("./dataset/validation", crop_box)
