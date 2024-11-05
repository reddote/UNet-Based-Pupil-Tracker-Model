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
            transforms.Normalize(mean=[0.7324], std=[0.1679])
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
        predicted_gaze = output.cpu().numpy().flatten()  # GPU'dan CPU'ya taşı ve numpy dizisine dönüştür
        print(f"Predicted Gaze (gaze_x, gaze_y, gaze_z): {predicted_gaze}")
        self.run_image(image_3channel, 186.118255, 160.316709, 190.016142, predicted_gaze[0], predicted_gaze[1], predicted_gaze[2])
        return predicted_gaze

    def run_image(self, image, center_x, center_y, radius, x, y, z):

        gaze_x = int(center_x + center_x * x * z)
        gaze_y = int(center_y + center_y * y * z)

        end_point = (gaze_x, gaze_y)

        startpoint_int = (int(center_x), int(center_y))

        print(f"image height: {image.shape[0]}, image width: {image.shape[1]}")

        cv2.circle(image, (gaze_x, gaze_y), 2, (255, 0, 255), -1)
        cv2.circle(image, (int(center_x), int(center_y)), 2, (255, 255, 0), -1)
        cv2.circle(image, (int(center_x), int(center_y)), int(radius), (0, 255, 255), 3)
        cv2.line(image, startpoint_int, end_point, (255, 255, 0), 1)
        cv2.imshow('Image with Center', image)
        print(f"Center X: {center_x}, Center Y: {center_y}, gaze X: {x}, gaze Y: {y}, gaze Z: {z}, radius: {radius} ")
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
