import logging
import math
import os
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2

import test_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup logging
file_handler = logging.FileHandler(filename='log.txt')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('Logs')

# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Main:
    def __init__(self):
        # Initializing the model and other settings
        self.model = test_model.TestModel().to(device)
        self.model.load_state_dict(torch.load('5_efficientnet_b0_regression_0.pth', map_location=device))
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 168)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run2(self, image_path):
        # Method for testing the model on a single sample
        # 1. Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image not found or unable to open: {image_path}")

        # 2. Convert the image to 3 channels
        image_3channel = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 3. Convert the image to PIL format
        pil_image = Image.fromarray(image_3channel)

        # 4. Transform the image
        input_tensor = self.transform(pil_image)
        input_tensor = input_tensor.unsqueeze(
            0)  # Reshape to the 4D tensor format expected by the model (batch_size, channels, height, width)

        # 5. Make a prediction using the model
        with torch.no_grad():  # Disable gradient computations
            input_tensor = input_tensor.to(device)
            output = self.model(input_tensor)

        # 6. Get the prediction and return it
        predicted_pupil = output.cpu().numpy().flatten()  # GPU'dan CPU'ya taşı ve numpy dizisine dönüştür
        print(f"Predicted Pupil Diameter (angle, center_x, center_y, width, height): {predicted_pupil}")
        self.run_image(image_3channel, predicted_pupil[0], predicted_pupil[1],
                       predicted_pupil[2], predicted_pupil[3], predicted_pupil[4])
        return predicted_pupil

    def run_image(self, image, center_x, center_y, angle, width, height):
        h, w, c = image.shape
        print(f'{h}, {w}, {c}')
        center = (center_x * w, center_y*h)
        axes_lengths = (width*w / 2, height*h / 2)
        angles = math.degrees(angle)
        # print(f"image height: {image.shape[0]}, image width: {image.shape[1]}")
        startpoint_int = (int(center[0]), int(center[1]))
        axes_int = (int(axes_lengths[0]), int(axes_lengths[1]))
        print(
            f"Center X: {center_x*w}, Center Y: {center_y*h}, Width: {width*w}, "
            f"Height : {height*h}, Angle : {angle} ")

        cv2.ellipse(image, startpoint_int, axes_int, angles, 0, 360, (255, 0, 0), 2)
        cv2.imshow('Image with Center', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example run
if __name__ == '__main__':
    main = Main()
    project_folder = os.path.dirname(__file__)
    image_path = os.path.join(project_folder, 'predict')
    image_path = os.path.join(image_path, '00010.jpg')
    test_image_path = image_path  # Test etmek istediğiniz resmin yolu
    main.run2(test_image_path)
