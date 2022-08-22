from multiprocessing import Pool, Manager, Queue, Process

# from queue import Queue
from threading import Thread
import threading
import time
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# def split(l, elements):
#     intervall_len = round(len(l) / (elements))
#     res = []
#     if len(l) % elements != 0:
#         # intervall_len = len(l) // (elements - 1)
#         res = [
#             l[i * intervall_len : (i + 1) * intervall_len] for i in range(elements - 1)
#         ] + [l[(elements - 1) * intervall_len :]]
#     else:
#         res = [l[i * intervall_len : (i + 1) * intervall_len] for i in range(elements)]

#     for i, intervall in enumerate(res):
#         res[i] = list(
#             zip(range(i * intervall_len, i * intervall_len + len(intervall)), intervall)
#         )
#     return res


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
        self.filenames = self.filenames[:100]
        self.filenames = list(map(lambda x: os.path.join(img_path, x), self.filenames))
        self.n_files = len(self.filenames)

        self.bg_size = 100
        self.median_bg_size = 5
        self.max_queue_size = 10

    def load_and_add(self, fns, sum_list):
        s = 0
        for fn in fns:
            img = cv.imread(fn, cv.IMREAD_GRAYSCALE).astype(np.float64)
            s += img
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
            queue.put(img)
            i += 1

    def preload_current_img(self, queue, start_index, end_index):
        i = start_index
        while i <= end_index:
            fn = self.filenames[i]
            img = cv.imread(fn)
            queue.put((cv.cvtColor(img, cv.COLOR_BGR2GRAY), img, fn.split("\\")[-1]))
            i += 1

    def detect(self, corrected, img, fn):
        thresh = cv.threshold(corrected, 0, 255, cv.THRESH_TRIANGLE)[1]
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        for cnt in cnts:
            if cv.contourArea(cnt) < 400:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            crop = img[y : y + h, x : x + w]
            # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # cv.imwrite(
        #     os.path.join(self.save_path, fn),
        #     img,
        # )

    def mean_segmenter(self, indices):
        first_index = indices[0]
        last_index = indices[-1]
        start = time.perf_counter()

        bg_sum, start_index = self.compute_initial_mean_bg(first_index)
        end_index = last_index + self.bg_size // 2
        end_index = self.n_files - 1 if end_index >= self.n_files else end_index

        last_img_queue = Queue(self.max_queue_size)
        next_img_queue = Queue(self.max_queue_size)
        current_img_queue = Queue(self.max_queue_size)

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

        for i in range(first_index, last_index + 1):
            if i > self.bg_size // 2 and i < self.n_files - self.bg_size // 2:
                last_img = last_img_queue.get(timeout=5)
                next_img = next_img_queue.get(timeout=5)
                bg_sum -= last_img
                bg_sum += next_img

            bg = bg_sum / self.bg_size
            bg = bg.astype(np.uint8)
            gray, current_img, fn = current_img_queue.get(timeout=5)
            corrected = cv.absdiff(gray, bg)

            while threading.active_count() >= self.n_threads:
                print("Waiting...")
                time.sleep(0.01)

            Thread(target=self.detect, args=(corrected, current_img, fn)).start()

            self.bg_mean_vals[i] = np.mean(bg)  # type: ignore
            self.img_mean_vals[i] = np.mean(gray)  # type: ignore

        end = time.perf_counter()
        total_duration = end - start
        print(f"bg finished in {total_duration: .2f}s")

    def main_mean_segmenter(self, cores, n_threads):

        self.manager = Manager()
        self.bg_mean_vals = self.manager.list([0 for _ in range(self.n_files)])
        self.img_mean_vals = self.manager.list([0 for _ in range(self.n_files)])

        self.n_threads = 8
        self.cores = self.n_files // (self.bg_size * 3)
        self.cores = 16 if self.cores > 16 else self.cores
        self.cores = 1 if self.cores < 1 else self.cores

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

    def compute_initial_median_bg(self, index):
        start_index = index - self.median_bg_size // 2
        start_index = 0 if start_index < 0 else start_index
        end_index = start_index + self.median_bg_size
        end_index = self.n_files - 1 if end_index >= self.n_files else end_index

        median_list = []
        for index in range(start_index, end_index + 1):
            img = cv.imread(self.filenames[index], cv.IMREAD_GRAYSCALE)
            median_list.append(img)

        return median_list, start_index, end_index

    def median_segmenter(self, indices):
        start = time.perf_counter()
        median_list, start_index, end_index = self.compute_initial_median_bg(indices[0])
        next_img_queue = Queue(self.max_queue_size)
        current_img_queue = Queue(self.max_queue_size)
        sliding_index = 0

        last_index = indices[-1] + self.median_bg_size // 2
        last_index = self.n_files - 1 if last_index >= self.n_files else last_index
        Thread(
            target=self.preload,
            args=(
                next_img_queue,
                end_index,
                last_index,
            ),
        ).start()

        Thread(
            target=self.preload_current_img,
            args=(current_img_queue, indices[0], indices[-1]),
        ).start()

        for i in indices:
            if (
                i > self.median_bg_size // 2
                and i < self.n_files - self.median_bg_size // 2
                and i != indices[0]
            ):

                next_img = next_img_queue.get(timeout=3)
                median_list[sliding_index] = next_img
                sliding_index += 1
                sliding_index = (
                    0 if sliding_index >= len(median_list) else sliding_index
                )

            bg = np.median(median_list, axis=0).astype(np.uint8)
            gray, current_img, fn = current_img_queue.get(timeout=5)
            corrected = cv.absdiff(gray, bg)

            while threading.active_count() >= self.n_threads:
                print("Waiting...")
                time.sleep(0.01)

            Thread(target=self.detect, args=(corrected, current_img, fn)).start()

            self.bg_mean_vals[i] = np.mean(bg)  # type: ignore
            self.img_mean_vals[i] = np.mean(gray)  # type: ignore

        end = time.perf_counter()
        total_duration = end - start
        print(f"bg finished in {total_duration: .2f}s")

    def main_median_segmenter(self, cores, n_threads):
        self.cores = cores
        self.n_threads = n_threads

        start = time.perf_counter()
        splitted_indices = split(list(range(self.n_files)), self.cores)

        print(f"Files: {self.n_files}; Cores: {self.cores}; Threads: {self.n_threads}")
        with Pool(self.cores) as pool:
            pool.map(self.median_segmenter, splitted_indices)

        end = time.perf_counter()
        total_duration = end - start

        print(
            f"Total time: {total_duration: .2f}s",
            f"Avg time per img: {total_duration / len(self.filenames): .2f}s",
        )

        plt.plot(self.img_mean_vals, "bo", label="Mean values of images")
        plt.plot(
            self.bg_mean_vals,
            label="Mean values of background images",
            color="orange",
        )
        plt.legend()
        plt.savefig(os.path.join(save_path, "development_of_mean_values_median.png"))
        plt.close()
        return total_duration

    def preload_median(self, queue, fns):
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

            for j in range(start_index, end_index):
                img = cv.imread(fns[j], cv.IMREAD_GRAYSCALE)
                imgs.append(img)

            queue.put((imgs, img_index))
            # print(i, len(imgs), img_index)

    def median_segmenter2(self, index):
        next_imgs, img_index = self.next_img_queue.get()
        img = next_imgs[img_index]
        bg = np.median(next_imgs, axis=0).astype(np.uint8)
        corrected = cv.absdiff(img, bg)
        self.detect(corrected, img, "")

    def main_median_segmenter2(self, cores):
        self.cores = cores

        start = time.perf_counter()

        splitted = split(self.filenames, 3)

        m = Manager()
        self.next_img_queue = m.Queue(5)
        for fns in splitted:
            p = Process(target=self.preload_median, args=(self.next_img_queue, fns))
            p.start()

        print(f"Files: {self.n_files}; Cores: {self.cores}")
        with Pool(self.cores) as pool:
            pool.map(self.median_segmenter2, range(self.n_files))

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
    img_path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
    save_path = "Results/Test_multiprocessing"

    # for f in os.listdir(save_path):
    #     os.remove(os.path.join(save_path, f))

    s = SegmenterMultiProcessing(img_path, save_path)
    duration = s.main_median_segmenter2(8)
    # cores_list.append(cores)
    # durations.append(duration)

    # plt.plot(
    #     cores_list, durations, "bo", label="Duration of segmentation for 1000 Images"
    # )
    # plt.legend()
    # plt.savefig(os.path.join(save_path, "benchmark_cores_median.png"))
    # plt.close()
