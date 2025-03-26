import os
import cv2  # OpenCV is required for video processing

# Define the source and destination paths
source_path = r"D:\videosData"
destination_path = r"C:\Users\malpe\Desktop\Project\eNet\newSeg"


# Function to create a unique folder for each video
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Initialize a counter for naming folders
counter = 1

# Walk through the source directory to find and process videos
for root, dirs, files in os.walk(source_path):
    for file in files:
        if file.endswith(".mp4pupil_seg_2D.mp4"):
            # Construct the full file path
            video_path = os.path.join(root, file)

            # Create a unique folder for this video
            video_folder = os.path.join(destination_path, file)
            create_folder(video_folder)

            # Open the video file
            video_capture = cv2.VideoCapture(video_path)

            frame_counter = 1  # Initialize the frame counter for naming
            success, frame = video_capture.read()

            while success:
                # Define the image file name (with leading zeros)
                image_name = f"{frame_counter:05d}.jpg"
                image_path = os.path.join(video_folder, image_name)

                # Save the current frame as an image
                cv2.imwrite(image_path, frame)

                # Read the next frame
                success, frame = video_capture.read()
                frame_counter += 1

            # Release the video capture object
            video_capture.release()
            print(f"Processed video: {video_path}, saved to: {video_folder}")

            # Increment the folder counter for the next video
            counter += 1

print("Video processing completed.")
