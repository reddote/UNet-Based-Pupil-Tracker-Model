import cv2
import os
import pandas as pd
import math


class FindGaze:
    def __init__(self):
        project_folder = os.path.dirname(__file__)
        project_folder_image = r"C:\Users\malpe\Desktop\Project\eNet"
        self.image_path = os.path.join(project_folder_image, 'output_frames\\1\\' + '00001.jpg')
        self.pupil_path = os.path.join(project_folder, 'videosData\\' + '1-IrisEli.csv')
        self.eyeball_path = os.path.join(project_folder, 'videosData\\' + '1-EyeBall.csv')
        self.pupil_in_iris_path = os.path.join(project_folder, 'videosData\\' + '1-PupilInIris.csv')
        self.validity_path = os.path.join(project_folder, 'videosData\\' + '1-Validity.csv')
        # self.video_path = os.path.join(project_folder, 'predict\\' + '2.mp4')
        self.image = cv2.imread(self.image_path)
        # self.video = cv2.VideoCapture(self.video_path)

    def run(self, item):
        pupil_csv = pd.read_csv(self.pupil_path)
        eyeball_csv = pd.read_csv(self.eyeball_path)
        pupil_in_iris_csv = pd.read_csv(self.pupil_in_iris_path)
        validity_csv = pd.read_csv(self.validity_path)

        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        center = (pupil_in_iris_csv.iloc[item]['CENTER X'], pupil_in_iris_csv.iloc[item]['CENTER Y'])
        axes_lengths = (pupil_in_iris_csv.iloc[item]['WIDTH']/2, pupil_in_iris_csv.iloc[item]['HEIGHT']/2)
        angles = pupil_in_iris_csv.iloc[item]['ANGLE']

        # # Calculate alpha and clip it within the range [-1, 1] to avoid domain errors
        # k = axes_lengths[0]/axes_lengths[1]
        # corrected_height = axes_lengths[1]*k
        # radius_angle = int((axes_lengths[0]/2)*4.5)
        # alpha = corrected_height / axes_lengths[0]
        # # Clip alpha to the range [-1, 1]
        # alpha = min(0.99, max(-0.99, alpha))
        # print(f"alpha = {radius_angle}")
        # theta = math.acos(alpha)
        # phi = math.radians(angles)
        # X_eye = center[0] + radius_angle * math.sin(theta) * math.cos(phi)
        # Y_eye = center[1] + radius_angle * math.sin(theta) * math.sin(phi)
        # Z_eye = radius_angle*math.sin(theta)

        # test_axes = (int(axes_lengths[0]), int(corrected_height))
        #
        # test_point = (int(X_eye), int(Y_eye))

        # print(f"3D Eye Location X: {X_eye} Y: {Y_eye} Z: {Z_eye}")

        # radius = int(eyeball_csv.iloc[item]['RADIUS'])
        # center_eye_ball = (int(eyeball_csv.iloc[item]['X']), int(eyeball_csv.iloc[item]['Y']))

        # print(f"image height: {image.shape[0]}, image width: {image.shape[1]}")
        height, width = image.shape
        startpoint_int = (int(center[0]), int(center[1]))
        axes_int = (int(axes_lengths[0]), int(axes_lengths[1]))
        print(f'height {height}, width {width}')
        print(f"Center X: {pupil_in_iris_csv.iloc[item]['CENTER X']}, Center Y: {pupil_in_iris_csv.iloc[item]['CENTER Y']}, Width: {pupil_in_iris_csv.iloc[item]['WIDTH']}, "
              f"Height : {pupil_in_iris_csv.iloc[item]['HEIGHT']}, Angle : {pupil_in_iris_csv.iloc[item]['ANGLE']} ")

        cv2.ellipse(image, startpoint_int, axes_int, angles, 0, 360, (255, 0, 0), 2)
        # cv2.ellipse(image, center_eye_ball, test_axes, radius_angle, 0, 360,  (255, 255, 255), 2)
        # cv2.circle(image, center_eye_ball, radius_angle, (0, 255, 0), 2)
        cv2.imshow('Image with Center', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


main = FindGaze()
main.run(0)

