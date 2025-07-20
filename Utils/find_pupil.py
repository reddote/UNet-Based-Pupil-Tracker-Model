import os
import sys

import file_controller
import cv2
import file_writer
import logging

# Setup logging
file_handler = logging.FileHandler(filename='../log.txt')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('Logs')


class FindPupil:
    def __init__(self):
        project_folder = os.path.dirname(__file__)
        self.file_name = 0
        self.image_path = os.path.join(project_folder, 'images')
        self.actual_image_path = os.path.join(self.image_path, self.file_name.__str__())
        self.csv_path = os.path.join(project_folder, 'NvGazeCSV')
        self.actual_csv_path = os.path.join(project_folder, self.file_name.__str__()+'.csv')
        self.fc = file_controller.FileController(self.actual_image_path)
        self.writer = file_writer.FileWriter(self.actual_csv_path)
        self.thresh_hold = [0, 70, 70, 80, 70, 120, 140, 70, 60, 80, 70, 135, 70, 28, 20, 20, 15, 20, 21, 15, 20]

    def next_image_folder(self):
        self.file_name += 1
        if self.file_name > 20:
            self.file_name = 0
            return
        # print(self.file_name)
        self.actual_image_path = os.path.join(self.image_path, self.file_name.__str__())
        self.actual_csv_path = os.path.join(self.csv_path, self.file_name.__str__()+'.csv')
        self.fc = file_controller.FileController(self.actual_image_path)
        self.writer = file_writer.FileWriter(self.actual_csv_path)

    def run(self):
        while True:
            self.next_image_folder()
            image_list = self.fc.get_files()
            print(image_list.__len__())
            print(self.thresh_hold[self.file_name])
            if len(image_list) == 0 or self.file_name == 0:
                break
            center_x = 0
            center_y = 0
            x1, x2, y1, y2 = 96, 544, 16, 464
            for images in image_list:
                image_name = images.removeprefix(self.actual_image_path + '\\')

                image = cv2.imread(images)

                roi = image[y1:y2, x1:x2]
                if image is None:
                    print('Could not load image')
                    exit()

                # Convert the image to grayscale
                gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Apply a binary threshold to the grayscale image
                ret, binary_image = cv2.threshold(gray_image, self.thresh_hold[self.file_name], 255, cv2.THRESH_BINARY)

                # Invert the binary image to make black spots black and everything else white
                inverted_image = cv2.bitwise_not(binary_image)

                # Find contours in the binary image
                contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    # Find the contour with the largest area, assuming it's the black circle
                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(image[y1:y2, x1:x2], largest_contour, -1, (0, 255, 255), 2)

                    # Calculate the moments of the largest contour
                    moments = cv2.moments(largest_contour)

                    # Calculate the center of the circle
                    if moments["m00"] != 0:
                        center_x = int(moments["m10"] / moments["m00"])
                        center_y = int(moments["m01"] / moments["m00"])
                    else:
                        logger.info(f"Pupil center not found {image_name}")
                        center_x, center_y = 0, 0

                    # Draw the center of the circle on the image
                    # cv2.circle(image, (center_x + x1, center_y + y1), 5, (0, 0, 255), -1)  # Draw a red dot at the center

                    # Display the result
                    # cv2.imshow('Image with Center', image)
                    print(f"Center of ({image_name}): ({center_x}, {center_y})")

                else:
                    logger.error(f"Contour not found {image_name}")
                if center_x == 0 and center_y == 0:
                    center_x = 0
                    center_y = 0
                else:
                    center_x = center_x + x1
                    center_y = center_y + y1

                self.writer.write(image_name, 'L', center_x, center_y)


find_pupil = FindPupil()
find_pupil.run()
