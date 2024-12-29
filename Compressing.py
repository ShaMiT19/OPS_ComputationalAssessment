# Load necessary libraries
import os
from PIL import Image
import re
from pathlib import Path

# Change these folder name accordingly

input_folder = "./TestSet"  # folder where all images are stored
output_folder = "./ResizedTestSet"  # folder where all resized images will be saved


i = 0  # image counter
for file in os.listdir(input_folder):  # reads all files in this directory
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        # if the file is in any of this format, proceed with resizing and renaming

        # Load filename
        filename = input_folder + "/" + file

        # Load image and resize it to a 100x100 RGB image
        y = Image.open(filename).convert("RGB").resize((299, 299))

        # Create new directory for resized images
        new_dir = Path(
            output_folder
        )  # new directory where altered images will be saved
        new_dir.mkdir(
            parents=True, exist_ok=True
        )  # BEWARE:it does not create a new dir if "dataset" already exists

        # Save resized images in jpg format
        new_file_name = file.split(".", 1)[0] + ".png"
        y.save(new_dir / new_file_name)

        i += 1  # increment image counter
