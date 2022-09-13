from multiprocessing import Pool, Manager
from segmenter_object import MeanSegmenter
from PIL import Image
import time
import os
import csv


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
    def __init__(
        self, img_path, save_path, save_full_imgs=False, save_crops=True, resize=None
    ):
        self.img_path = img_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.files = os.listdir(img_path)

        self.bg_size = 100
        self.median_bg_size = 5
        self.max_queue_size = 10
        self.save_crops = save_crops
        self.save_full_imgs = save_full_imgs

        manager = Manager()
        self.counter = manager.list([0, 0])

        self.min_area = 25
        self.min_var = 30

        self.meta_data = manager.list([])
        self.filtered_files = manager.list([])

        self.resize = resize
        self.shape = (5120, 5120)
        self.valid_endings = [".jpg", ".png", ".tif"]
        self.filter_files()

    def check_file(self, file):
        file = os.path.join(self.img_path, file)
        if file[-4:] in self.valid_endings and Image.open(file).size == self.shape:
            self.filtered_files.append(file)

    def filter_files(self):
        s = time.perf_counter()
        initial_n_files = len(self.files)
        with Pool() as pool:
            pool.map(self.check_file, self.files)
        self.files = list(self.filtered_files)
        self.n_files = len(self.files)
        print(
            f"Filtering done in {time.perf_counter() - s: .2f}s, {initial_n_files - self.n_files} files removed."
        )

    def start_segmenter(self, indices):
        segmenter = MeanSegmenter(
            self, indices[0], indices[-1], self.bg_size, self.n_threads
        )
        segmenter.segment()

    def main(self, cores, n_threads):
        start = time.perf_counter()
        self.n_threads = n_threads
        splitted_indices = split(range(self.n_files), cores)
        print(f"Files: {self.n_files}; Cores: {cores}; Threads: {n_threads}")

        with Pool(cores) as pool:
            pool.map(self.start_segmenter, splitted_indices)

        end = time.perf_counter()
        total_duration = end - start

        print(
            f"Total time: {total_duration: .2f}s",
            f"Avg time per img: {total_duration / len(self.files): .2f}s",
        )

        self.meta_data.sort()
        with open(os.path.join(self.save_path, "meta_data.csv"), "w") as f:
            writer = csv.writer(f, delimiter=";")
            for row in self.meta_data:
                writer.writerow(row)

        return total_duration


if __name__ == "__main__":
    img_path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
    save_path = "Results/Crops_Mean_Half_Resolution"

    for fn in os.listdir(save_path):
        os.remove(os.path.join(save_path, fn))

    s = SegmenterMultiProcessing(
        img_path,
        save_path,
        save_full_imgs=False,
        save_crops=True,  #  , resize=(2560, 2560)
    )
    s.main(cores=4, n_threads=8)
