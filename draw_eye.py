import cv2
import os
import pandas as pd


class FindGaze:
    def __init__(self):
        project_folder = os.path.dirname(__file__)
        project_folder_image = r"C:\Users\malpe\Desktop\Project\eNet"
        self.image_path = os.path.join(project_folder_image, 'output_frames\\1\\' + '00500.jpg')
        self.pupil_path = os.path.join(project_folder, 'videosData\\' + '1-Pupil.csv')
        self.pupil_in_iris_path = os.path.join(project_folder, 'videosData\\' + '1-PupilInIris.csv')
        self.validity_path = os.path.join(project_folder, 'videosData\\' + '1-Validity.csv')
        # self.video_path = os.path.join(project_folder, 'predict\\' + '2.mp4')
        self.image = cv2.imread(self.image_path)
        # self.video = cv2.VideoCapture(self.video_path)

    def run(self, item):
        pupil_csv = pd.read_csv(self.pupil_path)
        pupil_in_iris_csv = pd.read_csv(self.pupil_in_iris_path)
        validity_csv = pd.read_csv(self.validity_path)

        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        center = (pupil_csv.iloc[item]['CENTER X'], pupil_csv.iloc[item]['CENTER Y'])
        axes_lengths = (pupil_csv.iloc[item]['WIDTH']/2, pupil_csv.iloc[item]['HEIGHT']/2)
        angles = pupil_csv.iloc[item]['ANGLE']

        # print(f"image height: {image.shape[0]}, image width: {image.shape[1]}")

        startpoint_int = (int(center[0]), int(center[1]))
        axes_int = (int(axes_lengths[0]), int(axes_lengths[1]))

        print(f"Center X: {pupil_csv.iloc[item]['CENTER X']}, Center Y: {pupil_csv.iloc[item]['CENTER Y']}, Width: {pupil_csv.iloc[item]['WIDTH']}, "
              f"Height : {pupil_csv.iloc[item]['HEIGHT']}, Angle : {pupil_csv.iloc[item]['ANGLE']} ")

        cv2.ellipse(image, startpoint_int, axes_int, angles, 0, 360, (255, 0, 0), 2)
        cv2.imshow('Image with Center', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


main = FindGaze()
main.run(499)

