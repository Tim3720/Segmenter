from multiprocessing import Pool, Manager, Process

# from queue import Queue
from threading import Thread
import threading
import time
import os
import queue
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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
        self.filenames = os.listdir(img_path) * 2
        self.filenames = self.filenames[:1000]
        self.filenames = list(map(lambda x: os.path.join(img_path, x), self.filenames))
        self.n_files = len(self.filenames)

        self.bg_size = 100
        self.median_bg_size = 5
        self.max_queue_size = 10

        manager = Manager()
        self.bg_mean_vals = manager.list([0 for _ in range(self.n_files)])
        self.img_mean_vals = manager.list([0 for _ in range(self.n_files)])
        self.counter = manager.list([0])

        self.min_area = 400

    def load_and_add(self, fns, sum_list):
        s = 0
        for fn in fns:
            img = cv.imread(fn, cv.IMREAD_GRAYSCALE).astype(np.float64)
            img = cv.resize(img, (2000, 2000))
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
            img = cv.resize(img, (2000, 2000))
            queue.put(img)
            i += 1

    def preload_current_img(self, queue, start_index, end_index):
        i = start_index
        while i <= end_index:
            fn = self.filenames[i]
            img = cv.imread(fn)
            img = cv.resize(img, (2000, 2000))
            queue.put((cv.cvtColor(img, cv.COLOR_BGR2GRAY), img, fn.split("\\")[-1]))
            i += 1

    def detect(self, corrected, img, fn):
        thresh = cv.threshold(corrected, 0, 255, cv.THRESH_TRIANGLE)[1]
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        for cnt in cnts:
            if cv.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            crop_corr = corrected[y : y + h, x : x + w]
            crop = img[y : y + h, x : x + w]
            var = np.var(crop_corr)
            if var < 30:
                continue
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv.putText(
            #     img,
            #     str(var),
            #     (x, y),
            #     cv.FONT_HERSHEY_DUPLEX,
            #     1,
            #     (0, 255, 0),
            #     1,
            # )

        cv.imwrite(
            os.path.join(self.save_path, fn),
            img,
        )

        self.counter[0] += 1
        if (self.counter[0] / self.n_files * 100) % 10 == 0:
            print(f"{(self.counter[0] * 100 // self.n_files)}% done...")

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
            gray, current_img, fn = current_img_queue.get(timeout=5)
            corrected = cv.absdiff(gray, bg)

            while threading.active_count() >= self.n_threads:
                print("Waiting...", threading.enumerate())
                time.sleep(0.01)

            t = Thread(target=self.detect, args=(corrected, current_img, f"{i}.jpg"))
            t.start()
            threads.append(t)

            self.bg_mean_vals[i] = np.mean(bg)  # type: ignore
            self.img_mean_vals[i] = np.mean(gray)  # type: ignore

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
        return total_duration


if __name__ == "__main__":
    img_path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
    save_path = "Results/Test_multiprocessing_Mean"

    for fn in os.listdir(save_path):
        os.remove(os.path.join(save_path, fn))

    s = SegmenterMultiProcessing(img_path, save_path)
    duration = s.main_mean_segmenter(3, 5)
