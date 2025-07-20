import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
from torchvision.transforms import transforms
from PIL import Image
import math


class GazeDataset(Dataset):
    def __init__(self, phase='train', transform=None):
        self.phase = phase
        project_folder = os.path.dirname(__file__)
        # project_folder_images = r"C:\Users\malpe\Desktop\Project\eNet"
        self.image_path = os.path.join(project_folder, 'output_frames')
        self.segmentation_path = os.path.join(project_folder, 'segmentation')
        self.pupil_csv_path = os.path.join(project_folder, 'videosData')
        self.validity_csv_path = os.path.join(project_folder, 'videosData')
        # self.eye_csv_path = os.path.join(project_folder, 'videosData')

        self.train_file_number = [1, 2, 3, 4, 5, 6, 19, 25, 15, 16]
        self.val_file_number = [17, 18, 19]

        # Load different files based on the phase
        if phase == 'train':
            self.file_name = 0
            self.actual_image_path = os.path.join(self.image_path, str(self.train_file_number[self.file_name]))
            self.actual_segmentation_path = os.path.join(self.segmentation_path, str(self.train_file_number[self.file_name]))
            self.actual_pupil_path = os.path.join(self.pupil_csv_path, str(self.train_file_number[self.file_name]) + str('-Pupil') + '.csv')
            self.actual_validity_path = os.path.join(self.validity_csv_path, str(self.train_file_number[self.file_name]) + str('-Validity') + '.csv')
            # self.actual_eye_csv_path = os.path.join(self.eye_csv_path, str(self.file_name) + str('-Center') + '.csv')
            self.images_path_array = {}
            self.init_paths(self.train_file_number)
        elif phase == 'val':
            self.file_name = 0
            self.actual_image_path = os.path.join(self.image_path, str(self.val_file_number[self.file_name]))
            self.actual_segmentation_path = os.path.join(self.segmentation_path,
                                                         str(self.train_file_number[self.file_name]))
            self.actual_pupil_path = os.path.join(self.pupil_csv_path, str(self.val_file_number[self.file_name]) + str('-Pupil') + '.csv')
            self.actual_validity_path = os.path.join(self.validity_csv_path,
                                                     str(self.val_file_number[self.file_name]) + str('-Validity') + '.csv')
            self.images_path_array = {}
            self.init_paths(self.val_file_number)
            # self.actual_eye_csv_path = os.path.join(self.eye_csv_path, str(self.file_name) + str('-Center') + '.csv')

        self.transform = transform

    def __len__(self):
        return len(self.images_path_array)

    def init_paths(self, folder_number_array):
        folder = sorted(os.listdir(self.image_path), key=lambda x: int(x))
        counter = 0
        for folder_number in folder:
            if self.file_name == -1:
                break

            idx = -1
            # Ensure this is a list of paths
            folder_number = os.path.join(self.image_path, str(folder_number_array[self.file_name]))
            # Ensure this is a list all the path for the segmentation images
            folder_number_segmentation = os.path.join(self.segmentation_path, str(folder_number_array[self.file_name]))
            # print(f"Folder paths provided: {folder_number}")
            file_list = sorted(os.listdir(folder_number), key=lambda x: int(x.split('.')[0]))
            # print(f"file_list last{len(file_list)}")
            for file in file_list:
                file_path = os.path.join(folder_number, file)
                segmentation_path = os.path.join(folder_number_segmentation, file)
                self.images_path_array[counter] = (
                    file_path,
                    self.actual_pupil_path,
                    self.actual_validity_path,
                    idx,
                    segmentation_path,
                )
                #print(f"File name {self.file_name}:")
                #print(f"Entry {counter}:", self.images_path_array[counter])
                counter += 1
                idx += 1
            # print(f"self Gaze : {self.actual_gaze_csv_path}")
            self.next_data()

    # def init_paths2(self):
    #     folder = sorted(os.listdir(self.image_path), key=lambda x: int(x))
    #     counter = 0
    #     for folder_number in folder:
    #         # print(f"self Gaze : {self.actual_gaze_csv_path}")
    #         if self.file_name == -1:
    #             break
    #         print(folder_number)
    #         self.next_data()

    def __getitem__(self, item):
        while True:
            # dictionary 0 is file name
            # 1 is gaze.csv
            # 2 is movement.csv
            # print(f"img : {self.images_path_array[item][0]}")

            img_name = self.images_path_array[item][0]
            segmentation_name = self.images_path_array[item][4]
            # print(segmentation_name)
            temp_image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            segmentation_mask = cv2.imread(segmentation_name, cv2.IMREAD_GRAYSCALE)
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

            center_x = self.pupil_csv.iloc[idx]['CENTER X']/384
            center_y = self.pupil_csv.iloc[idx]['CENTER Y']/288
            angle = math.radians(self.pupil_csv.iloc[idx]['ANGLE'])
            width = self.pupil_csv.iloc[idx]['WIDTH']/384
            height = self.pupil_csv.iloc[idx]['HEIGHT']/288

            # Normalize mask to [0, 1] range
            mask = (np.float32(segmentation_mask) >= 200.0).astype(int)  # Pupil = 1, Non-pupil = 0

            # Prepare the three channels
            channel1 = torch.tensor(mask, dtype=torch.long).unsqueeze(0)  # Pupil region
            channel2 = torch.tensor(1 - mask, dtype=torch.float32).unsqueeze(0)  # Non-pupil region
            channel3 = torch.zeros_like(channel1)  # Trivial class (all zeros)
            # Combine the three channels into a single tensor
            segmentation_tensor = torch.cat([channel1, channel2, channel3], dim=0)  # Shape: [3, H, W]

            image_3channel = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image_3channel)

            if self.transform is not None:
                image = self.transform(image)

            # Convert transformed image to tensor
            # Normalize to [0, 1]
            # image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
            # input_tensor = image_tensor.unsqueeze(0)  # Shape becomes [1, 3, 288, 384]
            # print(input_tensor.shape)
            return image, channel1

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
        if self.phase == 'train':
            if self.file_name >= self.train_file_number.__len__():
                self.file_name = -1
                return
            self.actual_image_path = os.path.join(self.image_path, str(self.train_file_number[self.file_name]))
            self.actual_segmentation_path = os.path.join(self.segmentation_path,
                                                         str(self.train_file_number[self.file_name]))
            self.actual_pupil_path = os.path.join(self.pupil_csv_path,
                                                  str(self.train_file_number[self.file_name]) + str('-Pupil') + '.csv')
            self.actual_validity_path = os.path.join(self.validity_csv_path,
                                                     str(self.train_file_number[self.file_name]) + str(
                                                         '-Validity') + '.csv')
        else:
            if self.file_name >= self.val_file_number.__len__():
                self.file_name = -1
                return
            self.actual_image_path = os.path.join(self.image_path, str(self.val_file_number[self.file_name]))
            self.actual_segmentation_path = os.path.join(self.segmentation_path,
                                                         str(self.train_file_number[self.file_name]))
            self.actual_pupil_path = os.path.join(self.pupil_csv_path,
                                                  str(self.val_file_number[self.file_name]) + str('-Pupil') + '.csv')
            self.actual_validity_path = os.path.join(self.validity_csv_path,
                                                     str(self.val_file_number[self.file_name]) + str(
                                                         '-Validity') + '.csv')
        # print(f"Train file number {self.file_name}")
        # self.actual_eye_csv_path = os.path.join(self.eye_csv_path, str(self.file_name) + str('-Center') + '.csv')
        # print(f"next Data gaze : {self.actual_gaze_csv_path}")






dataset = GazeDataset(phase='train', transform=None)
dataset.__getitem__(0)
"""


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