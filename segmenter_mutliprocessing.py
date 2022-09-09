from multiprocessing import Pool, Manager, Process

from threading import Thread
import threading
import time
import os
import queue
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image


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
        # self.filenames = [os.path.join(img_path, file) for file in self.filenames]
        # self.n_files = len(self.filenames)

        self.bg_size = 100
        self.median_bg_size = 5
        self.max_queue_size = 10

        manager = Manager()
        # self.bg_mean_vals = manager.list([0 for _ in range(self.n_files)])
        # self.img_mean_vals = manager.list([0 for _ in range(self.n_files)])
        self.counter = manager.list([0, 0])
        self.next_img_queue_median = manager.Queue(5)

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

    def load_and_add(self, fns, sum_list):
        s = 0
        for fn in fns:
            img = cv.imread(fn, cv.IMREAD_GRAYSCALE).astype(np.float64)
            # img = cv.resize(img, (2560, 2560))
            s += img  # type: ignore
        sum_list.append(s)

    def compute_initial_mean_bg(self, index):
        start_index = index - self.bg_size // 2
        start_index = 0 if start_index < 0 else start_index
        end_index = start_index + self.bg_size
        end_index = self.n_files - 1 if end_index >= self.n_files else end_index

        bg_fns = split(self.filenames[start_index:end_index], self.n_threads)
        bg_sum = []
        threads = []
        for i, fns in enumerate(bg_fns):
            t = Thread(target=self.load_and_add, args=(fns, bg_sum))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        bg_sum = np.sum(bg_sum, axis=0)
        return bg_sum, start_index

    def preload(self, queue, start_index, end_index):
        i = start_index
        while i <= end_index:
            img = cv.imread(self.filenames[i], cv.IMREAD_GRAYSCALE)
            # img = cv.resize(img, (2560, 2560))
            queue.put(img)
            i += 1

    def preload_current_img(self, queue, start_index, end_index):
        i = start_index
        while i <= end_index:
            fn = self.filenames[i]
            img = cv.imread(fn)
            # img = cv.resize(img, (2560, 2560))
            queue.put((cv.cvtColor(img, cv.COLOR_BGR2GRAY), img, fn.split("\\")[-1]))
            i += 1

    def detect(self, corrected, img, fn):

        thresh = cv.threshold(corrected, 0, 255, cv.THRESH_TRIANGLE)[1]
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        counter = 0

        rects, fns = [], []

        for cnt in cnts:
            if cv.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            crop_corr = corrected[y : y + h, x : x + w]
            crop = img[y : y + h, x : x + w]
            var = np.var(crop_corr)
            if var < self.min_var:
                continue

            counter += 1
            crop_fn = fn[:-4] + f"_{counter}.jpg"

            rects.append((x, y, w, h))
            fns.append(crop_fn)

            cv.imwrite(os.path.join(self.save_path, crop_fn), crop)
        #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # cv.imwrite(
        #     os.path.join(self.save_path, fn),
        #     img,
        # )

        self.save_info(rects, fns, fn)

        self.counter[0] += 1
        p = (self.counter[0] / self.n_files * 100) // 10 * 10
        if p > self.counter[1]:
            self.counter[1] = p  # type: ignore
            print(f"{(self.counter[0] * 100 // self.n_files)}% done...")

    def save_info(self, rects, fns, fn):
        with open(os.path.join(self.save_path, fn[:-4] + ".csv"), "w") as f:
            writer = csv.writer(f, delimiter=";")
            x = [rect[0] for rect in rects]
            y = [rect[1] for rect in rects]
            w = [rect[2] for rect in rects]
            h = [rect[3] for rect in rects]

            data = zip(x, y, w, h, fns)
            for row in data:
                writer.writerow(row)

    def mean_segmenter(self, indices):
        first_index = indices[0]
        last_index = indices[-1]
        start = time.perf_counter()

        bg_sum, start_index = self.compute_initial_mean_bg(first_index)
        end_index = last_index + self.bg_size // 2
        end_index = self.n_files if end_index >= self.n_files else end_index
        end_index -= 1

        last_img_queue = queue.Queue(self.max_queue_size)
        next_img_queue = queue.Queue(self.max_queue_size)
        current_img_queue = queue.Queue(self.max_queue_size)

        Thread(
            target=self.preload,
            args=(last_img_queue, start_index, end_index - self.bg_size),
        ).start()

        Thread(
            target=self.preload,
            args=(
                next_img_queue,
                start_index + self.bg_size,
                end_index,
            ),
        ).start()

        Thread(
            target=self.preload_current_img,
            args=(current_img_queue, first_index, last_index),
        ).start()

        threads = []
        meta_data = []
        for i in range(first_index, last_index + 1):
            if (
                i > self.bg_size // 2
                and i < self.n_files - self.bg_size // 2
                and i != first_index
            ):
                last_img = last_img_queue.get(timeout=5)
                next_img = next_img_queue.get(timeout=5)
                bg_sum -= last_img
                bg_sum += next_img

            bg = bg_sum / self.bg_size
            bg = bg.astype(np.uint8)
            gray, img, fn = current_img_queue.get(timeout=5)

            corrected = cv.absdiff(gray, bg)

            img_mean = np.mean(gray)
            # img_std = np.std(gray)
            meta_data.append((i, img_mean, fn))

            neutral_bg = np.ones(bg.shape, np.uint8)  # type: ignore
            neutral_bg = neutral_bg * np.mean(bg)
            img = cv.absdiff(neutral_bg, corrected.astype(np.float64))
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

            while threading.active_count() >= self.n_threads:
                print("Waiting...", threading.enumerate())
                time.sleep(0.01)

            t = Thread(target=self.detect, args=(corrected, img, os.path.split(fn)[-1]))
            t.start()
            threads.append(t)

            # self.bg_mean_vals[i] = np.mean(bg)  # type: ignore
            # self.img_mean_vals[i] = np.mean(gray)  # type: ignore

        self.meta_data.extend(meta_data)

        for t in threads:
            t.join()

        end = time.perf_counter()
        total_duration = end - start
        print(f"bg finished in {total_duration: .2f}s")

    def main_mean_segmenter(self, cores, n_threads):
        self.cores = cores
        self.n_threads = n_threads

        start = time.perf_counter()
        splitted_indices = split(range(self.n_files), self.cores)

        print(f"Files: {self.n_files}; Cores: {self.cores}; Threads: {self.n_threads}")
        with Pool(self.cores) as pool:
            pool.map(self.mean_segmenter, splitted_indices)

        end = time.perf_counter()
        total_duration = end - start

        print(
            f"Total time: {total_duration: .2f}s",
            f"Avg time per img: {total_duration / len(self.filenames): .2f}s",
        )
        # plt.plot(self.img_mean_vals, "bo", label="Mean values of images")
        # plt.plot(
        #     self.bg_mean_vals,
        #     label="Mean values of background images",
        #     color="orange",
        # )
        # plt.legend()
        # plt.savefig(os.path.join(save_path, "development_of_mean_values.png"))
        # plt.close()

        self.meta_data.sort()
        with open(os.path.join(self.save_path, "meta_data.csv"), "w") as f:
            writer = csv.writer(f, delimiter=";")
            for row in self.meta_data:
                writer.writerow(row)

        return total_duration

    def preload_median(self, fns):
        for i in range(len(fns)):
            imgs = []
            if i <= self.median_bg_size // 2:
                start_index = 0
                img_index = i
                end_index = self.median_bg_size
            elif i >= len(fns) - self.median_bg_size // 2:
                start_index = len(fns) - self.median_bg_size
                img_index = i - len(fns)
                end_index = len(fns) - 1
            else:
                start_index = i - self.median_bg_size // 2
                end_index = i + self.median_bg_size // 2 + 1
                img_index = self.median_bg_size // 2 + 1

            filenames = []
            for j in range(start_index, end_index):
                img = cv.imread(fns[j], cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (2560, 2560))
                imgs.append(img)
                filenames.append(fns[j])
            fn = filenames[img_index]
            self.next_img_queue_median.put((imgs, fn))

    def median_segmenter(self, index):
        next_imgs, fn = self.next_img_queue_median.get()
        img = cv.imread(fn)
        # img = cv.resize(img, (2560, 2560))
        bg = np.median(next_imgs, axis=0).astype(np.uint8)
        bg = cv.resize(bg, (5120, 5120))
        corrected = cv.absdiff(cv.cvtColor(img, cv.COLOR_BGR2GRAY), bg)

        neutral_bg = np.ones(bg.shape, np.uint8) * np.mean(bg)  # type: ignore
        img = (neutral_bg - corrected).astype(np.uint8)

        self.detect(corrected, img, fn.split("\\")[-1])

    def main_median_segmenter(self, cores):
        self.cores = cores

        start = time.perf_counter()

        splitted = split(self.filenames, 5)

        for fns in splitted:
            p = Process(target=self.preload_median, args=(fns,))
            p.start()

        print(f"Files: {self.n_files}; Cores: {self.cores}")
        with Pool(self.cores) as pool:
            pool.map(self.median_segmenter, range(self.n_files))

        end = time.perf_counter()
        total_duration = end - start

        print(
            f"Total time: {total_duration: .2f}s",
            f"Avg time per img: {total_duration / len(self.filenames): .2f}s",
        )

        # plt.plot(self.img_mean_vals, "bo", label="Mean values of images")
        # plt.plot(
        #     self.bg_mean_vals,
        #     label="Mean values of background images",
        #     color="orange",
        # )
        # plt.legend()
        # plt.savefig(os.path.join(save_path, "development_of_mean_values_median.png"))
        # plt.close()
        return total_duration


if __name__ == "__main__":
    img_path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"  # Testprofil-M181-CTD-035-JPG
    save_path = "Results/Crops_Mean_Half_Resolution"

    # for fn in os.listdir(save_path):
    #     os.remove(os.path.join(save_path, fn))

    s = SegmenterMultiProcessing(img_path, save_path)
    duration = s.main_mean_segmenter(cores=3, n_threads=8)
    # duration = s.main_median_segmenter(12)
