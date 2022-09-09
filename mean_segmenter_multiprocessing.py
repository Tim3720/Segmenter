from multiprocessing import Pool, Manager, Process
from segmenter_object import MeanSegmenter
from PIL import Image
import time
import os


def split(l, elements):
    intervall_len = round(len(l) / (elements))
    res = []
    if len(l) % elements != 0:
        # intervall_len = len(l) // (elements - 1)
        res = [
            l[i * intervall_len : (i + 1) * intervall_len] for i in range(elements - 1)
        ] + [l[(elements - 1) * intervall_len :]]
    else:
        res = [l[i * intervall_len : (i + 1) * intervall_len] for i in range(elements)]
    return res


class SegmenterMultiProcessing:
    def __init__(self, img_path, save_path):
        self.img_path = img_path
        self.save_path = save_path
        self.filenames = os.listdir(img_path)

        self.bg_size = 100
        self.median_bg_size = 5
        self.max_queue_size = 10

        manager = Manager()
        self.counter = manager.list([0, 0])

        self.min_area = 25
        self.min_var = 30

        self.meta_data = manager.list([])
        self.filtered_files = manager.list([])

        self.shape = (5120, 5120)
        self.valid_endings = [".jpg", ".png", ".tif"]
        self.filter_files()

    def check_file(self, file):
        file = os.path.join(self.img_path, file)
        if file[-4:] in self.valid_endings and Image.open(file).size == self.shape:
            self.filtered_files.append(file)

    def filter_files(self):
        s = time.perf_counter()
        initial_n_files = len(self.filenames)
        with Pool() as pool:
            pool.map(self.check_file, self.filenames)
        self.filenames = list(self.filtered_files)
        self.n_files = len(self.filenames)
        print(
            f"Filtering done in {time.perf_counter() - s: .2f}s, {initial_n_files - self.n_files} files removed."
        )