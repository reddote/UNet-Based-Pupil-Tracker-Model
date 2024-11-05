import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
from torchvision.transforms import transforms
from PIL import Image


class GazeDataset(Dataset):
    def __init__(self, phase='train', transform=None):
        self.phase = phase
        project_folder = os.path.dirname(__file__)
        project_folder_images = r"C:\Users\malpe\Desktop\Project\eNet"
        self.image_path = os.path.join(project_folder_images, 'output_frames')
        self.pupil_csv_path = os.path.join(project_folder, 'videosData')
        self.validity_csv_path = os.path.join(project_folder, 'videosData')
        # self.eye_csv_path = os.path.join(project_folder, 'videosData')

        # Load different files based on the phase
        if phase == 'train':
            self.file_name = 1
            self.border = 5
            self.actual_image_path = os.path.join(self.image_path, str(self.file_name))
            self.actual_pupil_path = os.path.join(self.pupil_csv_path, str(self.file_name) + str('-Pupil') + '.csv')
            self.actual_validity_path = os.path.join(self.validity_csv_path, str(self.file_name) + str('-Validity') + '.csv')
            # self.actual_eye_csv_path = os.path.join(self.eye_csv_path, str(self.file_name) + str('-Center') + '.csv')
        elif phase == 'val':
            self.file_name = 6
            self.border = 8
            self.actual_image_path = os.path.join(self.image_path, str(self.file_name))
            self.actual_pupil_path = os.path.join(self.pupil_csv_path, str(self.file_name) + str('-Pupil') + '.csv')
            self.actual_validity_path = os.path.join(self.validity_csv_path,
                                                     str(self.file_name) + str('-Validity') + '.csv')
            # self.actual_eye_csv_path = os.path.join(self.eye_csv_path, str(self.file_name) + str('-Center') + '.csv')

        self.images_path_array = {}
        self.init_paths()

        self.transform = transform

    def __len__(self):
        return len(self.images_path_array)

    def init_paths(self):
        folder = sorted(os.listdir(self.image_path), key=lambda x: int(x))
        counter = 1
        for folder_number in folder:
            if self.file_name == 0:
                break

            idx = 0
            folder_number = os.path.join(self.image_path, str(self.file_name))  # Ensure this is a list of paths
            # print(f"Folder paths provided: {folder_number}")
            file_list = sorted(os.listdir(folder_number), key=lambda x: int(x.split('.')[0]))
            # print(f"file_list last{len(file_list)}")
            for file in file_list:
                file_path = os.path.join(folder_number, file)
                self.images_path_array[counter] = (
                    file_path,
                    self.actual_pupil_path,
                    self.actual_validity_path,
                    idx,
                )
                #print(f"File name {self.file_name}:")
                #print(f"Entry {counter}:", self.images_path_array[counter])
                counter += 1
                idx += 1
            # print(f"self Gaze : {self.actual_gaze_csv_path}")
            self.next_data()

    def __getitem__(self, item):
        while True:
            # dictionary 0 is file name
            # 1 is gaze.csv
            # 2 is movement.csv
            # print(f"img : {self.images_path_array[item][0]}")

            img_name = self.images_path_array[item][0]
            # print(img_name)
            temp_image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            # Load the CSV files
            self.pupil_csv = pd.read_csv(self.images_path_array[item][1])
            self.validity = pd.read_csv(self.images_path_array[item][2])
            idx = self.images_path_array[item][3]
            """
            # self.eye_csv = pd.read_csv(self.images_path_array[item][4])
            self.run_image(temp_image, self.pupil_csv.iloc[idx]['CENTER X'], self.pupil_csv.iloc[idx]['CENTER Y'],
                           self.pupil_csv.iloc[idx]['ANGLE'], self.pupil_csv.iloc[idx]['WIDTH'], self.pupil_csv.iloc[idx]['HEIGHT'])
            print(str(idx)+"/" + str(len(self.pupil_csv)))
            """
            if temp_image is None:
                print(f"Warning: Image not found or unable to open {item} : {temp_image}")
                item = (item + 1) % len(self)
                continue

            if idx < len(self.validity):
                pupil_error = self.validity.iloc[idx]['VALIDITY']
            else:
                break

            if pupil_error == -1:
                # print(f"Pupil center not detected or some error: {item}")
                item = (item + 1) % len(self)
                continue

            center_x = self.pupil_csv.iloc[idx]['CENTER X']
            center_y = self.pupil_csv.iloc[idx]['CENTER Y']
            angle = self.pupil_csv.iloc[idx]['ANGLE']
            width = self.pupil_csv.iloc[idx]['WIDTH']
            height = self.pupil_csv.iloc[idx]['HEIGHT']

            image_3channel = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image_3channel)

            if self.transform is not None:
                image = self.transform(image)

            return image, torch.tensor([center_x, center_y, angle, width, height], dtype=torch.float32)

    def run_image(self, image, center_x, center_y, angle, width, height):

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

    def next_data(self):
        self.file_name += 1
        if self.file_name > self.border:
            self.file_name = 0
            return
        # print(f"Train file number {self.file_name}")

        self.actual_image_path = os.path.join(self.image_path, str(self.file_name))
        self.actual_pupil_path = os.path.join(self.pupil_csv_path, str(self.file_name) + str('-Pupil') + '.csv')
        self.actual_validity_path = os.path.join(self.validity_csv_path,
                                                 str(self.file_name) + str('-Validity') + '.csv')
        # self.actual_eye_csv_path = os.path.join(self.eye_csv_path, str(self.file_name) + str('-Center') + '.csv')

        # print(f"next Data gaze : {self.actual_gaze_csv_path}")



"""
dataset = GazeDataset(phase='train', transform=None)
dataset.__getitem__(450)
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Veri kümesini oluşturma
dataset = GazeDataset(transform=transforms)

# DataLoader oluşturma
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    inputs, labels = batch  # batch'teki verilerin unpack edilmesi
    print(inputs.shape)  # Giriş verilerini yazdırır
    print(labels.shape)  # Etiketleri yazdırır
    break
"""