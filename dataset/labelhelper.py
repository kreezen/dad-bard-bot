import os


class LabelHelper:
    def __init__(self, label_file):
        self.label_file = label_file

    def label(self, label, end_range, start_range=0, name="screenshot_", extension=".png"):
        try:
            with open(self.label_file, 'a') as file:
                for i in range(start_range, end_range + 1):
                    print(i)
                    line = f"{name + str(i) + extension} {label}\n"
                    file.write(line)
        except FileNotFoundError:
            print("Error: The specified file or directory does not exist.")
        except PermissionError:
            print("Error: You do not have permission to write to this file.")
        except IOError as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        else:
            print("Writing to the file was successful.")

    def check_files_presence(self, root_dir):
        missing_files = []

        # Read the labels from the label file
        with open(self.label_file, "r") as file:
            for line in file:
                img_name, label = line.strip().split()

                # Check if the file exists in the specified directory
                img_path = os.path.join(root_dir, img_name)
                if not os.path.exists(img_path):
                    missing_files.append(img_name)

        if len(missing_files) == 0:
            print("All files mentioned in labels.txt are present.")
        else:
            print("Missing files:")
            for file in missing_files:
                print(file)


def main():

    # LabelHelper(label_file="./dataset/labels.txt").label(label="Trigger", start_range=97, end_range=144)

    # Example usage:
    root_dir = 'dataset/vali2'  # Replace with the actual directory path

    # Replace with the actual label file path
    label_file = 'dataset/vali2/labels.txt'

    # Check if all files mentioned in labels.txt are present
    LabelHelper(
        label_file="dataset/vali2/labels.txt").check_files_presence(root_dir)


if __name__ == "__main__":
    main()
