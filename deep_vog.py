import deepvog
import cv2

# Load the pre-trained network
model = deepvog.load_DeepVOG()

# Camera parameters (example values, replace with your actual camera specs)
focal_length = 4.15  # in mm (adjust according to your camera specs)
sensor_size = (6.0, 4.0)  # in mm (width, height, adjust accordingly)
video_shape = (1920, 1080)  # resolution of your input images/videos (width, height)

# Initialize the inferer
inferer = deepvog.gaze_inferer(model, focal_length, video_shape, sensor_size)

# Load and process a single image
image_path = "test_image.jpg"  # Path to your test image
image = cv2.imread(image_path)

# Ensure the image is in the correct format for the model
processed_image = deepvog.preprocess_image(image, video_shape)

# Infer gaze on the single image
gaze_result = inferer.infer(processed_image)

# Visualize the result
# Assuming `gaze_result` contains coordinates or visualization overlays
for point in gaze_result['landmarks']:
    cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

# Show the result using OpenCV
cv2.imshow("Gaze Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
