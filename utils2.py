import os
import shutil
import re

# Define the source and destination paths
source_path = r"D:\videosData"
destination_path = r"C:\Users\malpe\Desktop\Project\eNetPupilDetection\videosData"
checker_path = r"C:\Users\malpe\Desktop\checker"

# Function to split strings into segments of numbers and text for natural sorting
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Get folder names from the 'checker' path and sort them naturally
checker_folders = [f for f in os.listdir(checker_path) if os.path.isdir(os.path.join(checker_path, f))]
checker_folders.sort(key=natural_sort_key)

# Ensure the destination directory exists
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

# Initialize a counter for naming
counter = 1

# Walk through the source directory to find and copy files
for root, dirs, files in os.walk(source_path):
    files.sort(key=natural_sort_key)  # Ensure files are sorted naturally
    for file in files:
        if file.endswith(".mp4validity_pupil.csv"):
            # Check if the file name includes any of the folder names from 'checker'
            if any(folder in file for folder in checker_folders):
                # Construct full file paths
                source_file = os.path.join(root, file)
                new_file_name = f"{counter}-Validity.csv"
                destination_file = os.path.join(destination_path, new_file_name)

                # Copy the file with the new name
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")

                # Increment the counter
                counter += 1

print("File copying completed.")
