import logging
import os
import sys
import efficient_b0
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2


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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7324], std=[0.1679])
])


class Main:
    def __init__(self):
        # Model ve diğer ayarları başlatma
        self.model = efficient_b0.EfficientNetB0Regression().to(device)
        self.model.load_state_dict(torch.load('efficientnet_b0_regression_0.pth', map_location=device))
        self.model.eval()  # Modeli değerlendirme moduna al
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run2(self, image_path):
        # Modeli tek bir örnek üzerinde test eden yöntem
        # 1. Görüntüyü yükle
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image not found or unable to open: {image_path}")

        # 2. Görüntüyü 3 kanallı hale getir
        image_3channel = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 4. Görüntüyü PIL formatına çevir
        pil_image = Image.fromarray(image_3channel)

        # 5. Görüntüyü dönüştür
        input_tensor = self.transform(pil_image)
        input_tensor = input_tensor.unsqueeze(
            0)  # Modelin beklediği 4 boyutlu tensör şekline getir (batch_size, channels, height, width)

        # 6. Modeli kullanarak tahmin yap
        with torch.no_grad():  # Gradient hesaplamalarını kapat
            input_tensor = input_tensor.to(device)
            output = self.model(input_tensor)

        # 7. Tahmini al ve döndür
        predicted_pupil = output.cpu().numpy().flatten()  # GPU'dan CPU'ya taşı ve numpy dizisine dönüştür
        print(f"Predicted Pupil Diameter (angle, center_x, center_y, width, height): {predicted_pupil}")
        self.run_image(image_3channel, predicted_pupil[0], predicted_pupil[1],
                       predicted_pupil[2], predicted_pupil[3], predicted_pupil[4])
        return predicted_pupil

    def run_image(self, image, angle, center_x, center_y, width, height):
        center = (center_x, center_y)
        axes_lengths = (width / 2, height / 2)
        angles = angle

        # print(f"image height: {image.shape[0]}, image width: {image.shape[1]}")
        startpoint_int = (int(center[0]), int(center[1]))
        axes_int = (int(axes_lengths[0]), int(axes_lengths[1]))

        print(
            f"Center X: {center_x}, Center Y: {center_y}, Width: {width}, "
            f"Height : {height}, Angle : {angle} ")

        cv2.ellipse(image, startpoint_int, axes_int, angles, 0, 360, (255, 0, 0), 2)
        cv2.imshow('Image with Center', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Örnek kullanım
if __name__ == '__main__':
    main = Main()
    project_folder = os.path.dirname(__file__)
    image_path = os.path.join(project_folder, 'predict')
    image_path = os.path.join(image_path, '15501.jpg')
    test_image_path = image_path  # Test etmek istediğiniz resmin yolu
    main.run2(test_image_path)
