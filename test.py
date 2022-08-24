from multiprocessing import Pool, Process, Queue
from threading import Thread
import time
import os
import cv2 as cv
import numpy as np


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

    for i, intervall in enumerate(res):
        res[i] = list(
            zip(range(i * intervall_len, i * intervall_len + len(intervall)), intervall)
        )
    return res


def split2(l, elements):
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
        self.filenames = os.listdir(img_path)[:100]
        self.filenames = list(map(lambda x: os.path.join(img_path, x), self.filenames))

        self.bg_size = 5
        self.n_threads = 1
        self.cores = 16

    def load_imgs(self, fns):
        imgs = []
        grays = []
        for fn in fns:
            img = cv.imread(fn)
            if not img is None:
                imgs.append(img)
                grays.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        return imgs, grays

    def compute_median_bg(self, fns):
        imgs, grays = self.load_imgs(fns)
        bg = np.median(grays, axis=0)
        bg = bg.astype(np.uint8)
        return bg, imgs, grays

    def correct_img(self, index):
        if index < self.bg_size // 2:
            bg_fns = self.filenames[: self.bg_size]
            img_index = index
        elif index >= len(self.filenames) - self.bg_size // 2:
            bg_fns = self.filenames[-self.bg_size :]
            img_index = index - len(self.filenames)
        else:
            bg_fns = self.filenames[
                index - self.bg_size // 2 : index + self.bg_size // 2 + 1
            ]
            img_index = self.bg_size // 2 + self.bg_size % 2

        bg, imgs, grays = self.compute_median_bg(bg_fns)
        img = imgs[img_index]
        gray = grays[img_index]
        corrected = cv.absdiff(gray, bg)
        return corrected, img

    def detect(self, cnts, img, index):
        for cnt in cnts:
            if cv.contourArea(cnt) < 400:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv.imwrite(os.path.join(self.save_path, f"{index}.jpg"), img)

    def segment(self, index, filename):
        start = time.perf_counter()
        corrected, img = self.correct_img(index)
        thresh = cv.threshold(corrected, 0, 255, cv.THRESH_TRIANGLE)[1]
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        self.detect(cnts, img, index)
        end = time.perf_counter()
        duration = end - start
        return filename, duration

    def main(self):
        start = time.perf_counter()
        with Pool() as pool:
            results = pool.starmap(self.segment, enumerate(self.filenames))
            for filename, duration in results:
                print(f"{filename} completed in {duration: .2f}s")

        end = time.perf_counter()
        total_duration = end - start
        print(
            f"Total time: {total_duration: .2f}s",
            f"Avg time per img: {total_duration / len(self.filenames): .2f}s",
        )


if __name__ == "__main__":
    img_path = "C:/Users/timka/Documents/Arbeit/Peru"
    save_path = "Results/Test_multiprocessing"

    for f in os.listdir(save_path):
        os.remove(os.path.join(save_path, f))

    s = SegmenterMultiProcessing(
        img_path,
        save_path,
    )
    s.main()

    # def compute_initial_median_bg(self, index):
    #     start_index = index - self.median_bg_size // 2
    #     start_index = 0 if start_index < 0 else start_index
    #     end_index = start_index + self.median_bg_size
    #     end_index = self.n_files - 1 if end_index >= self.n_files else end_index

    #     median_list = []
    #     for index in range(start_index, end_index + 1):
    #         img = cv.imread(self.filenames[index], cv.IMREAD_GRAYSCALE)
    #         median_list.append(img)

    #     return median_list, start_index, end_index

    # def median_segmenter(self, indices):
    #     start = time.perf_counter()
    #     median_list, start_index, end_index = self.compute_initial_median_bg(indices[0])
    #     next_img_queue = Queue(self.max_queue_size)
    #     current_img_queue = Queue(self.max_queue_size)
    #     sliding_index = 0

    #     last_index = indices[-1] + self.median_bg_size // 2
    #     last_index = self.n_files - 1 if last_index >= self.n_files else last_index
    #     Thread(
    #         target=self.preload,
    #         args=(
    #             next_img_queue,
    #             end_index,
    #             last_index,
    #         ),
    #     ).start()

    #     Thread(
    #         target=self.preload_current_img,
    #         args=(current_img_queue, indices[0], indices[-1]),
    #     ).start()

    #     for i in indices:
    #         if (
    #             i > self.median_bg_size // 2
    #             and i < self.n_files - self.median_bg_size // 2
    #             and i != indices[0]
    #         ):

    #             next_img = next_img_queue.get(timeout=3)
    #             median_list[sliding_index] = next_img
    #             sliding_index += 1
    #             sliding_index = (
    #                 0 if sliding_index >= len(median_list) else sliding_index
    #             )

    #         bg = np.median(median_list, axis=0).astype(np.uint8)
    #         gray, current_img, fn = current_img_queue.get(timeout=5)
    #         corrected = cv.absdiff(gray, bg)

    #         while threading.active_count() >= self.n_threads:
    #             print("Waiting...")
    #             time.sleep(0.01)

    #         Thread(target=self.detect, args=(corrected, current_img, fn)).start()

    #         self.bg_mean_vals[i] = np.mean(bg)  # type: ignore
    #         self.img_mean_vals[i] = np.mean(gray)  # type: ignore

    #     end = time.perf_counter()
    #     total_duration = end - start
    #     print(f"bg finished in {total_duration: .2f}s")

    # def main_median_segmenter(self, cores, n_threads):
    #     self.cores = cores
    #     self.n_threads = n_threads

    #     start = time.perf_counter()
    #     splitted_indices = split(list(range(self.n_files)), self.cores)

    #     print(f"Files: {self.n_files}; Cores: {self.cores}; Threads: {self.n_threads}")
    #     with Pool(self.cores) as pool:
    #         pool.map(self.median_segmenter, splitted_indices)

    #     end = time.perf_counter()
    #     total_duration = end - start

    #     print(
    #         f"Total time: {total_duration: .2f}s",
    #         f"Avg time per img: {total_duration / len(self.filenames): .2f}s",
    #     )

    #     plt.plot(self.img_mean_vals, "bo", label="Mean values of images")
    #     plt.plot(
    #         self.bg_mean_vals,
    #         label="Mean values of background images",
    #         color="orange",
    #     )
    #     plt.legend()
    #     plt.savefig(os.path.join(save_path, "development_of_mean_values_median.png"))
    #     plt.close()
    #     return total_duration
