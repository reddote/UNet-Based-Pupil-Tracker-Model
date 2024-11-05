import csv
import os


class FileWriter:
    def __init__(self, path):
        self.path = path

    def write(self, image_file, eye, center_x, center_y):
        location_dict = [{'imagefile': image_file, 'eye': eye, 'Cx': center_x, 'Cy': center_y}]
        fields = ['imagefile', 'eye', 'Cx', 'Cy']
        file_exists = os.path.isfile(self.path)
        with open(self.path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerows(location_dict)
