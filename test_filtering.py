import os
import cv2 as cv
from multiprocessing import Manager, Pool
import time
from PIL import Image


class Test:
    def __init__(self, path) -> None:
        self.path = path
        self.files = os.listdir(path)
        manager = Manager()
        self.filtered_files = manager.list([])

    def check_file(self, file):
        shape = (5120, 5120)
        if (
            file.endswith(".jpg")
            and Image.open(os.path.join(self.path, file)).size == shape
        ):
            self.filtered_files.append(os.path.join(self.path, file))
            pass

    def main(self):
        print(len(self.files))
        s = time.perf_counter()
        with Pool() as pool:
            pool.map(self.check_file, self.files)
        print(f"{time.perf_counter() - s}s")
        print(len(self.filtered_files))


if __name__ == "__main__":
    path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
    t = Test(path)
    t.main()
