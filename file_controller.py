import glob


class FileController:
    def __init__(self, path):
        self.path = path

    def get_files(self):
        image_list = []
        for file in glob.glob(glob.escape(self.path) + "/*.jpg"):
            # print(file)
            image_list.append(file)
        return image_list
