import cv2 as cv
import numpy as np
import threading
import os
import time
from queue import Queue
import matplotlib.pyplot as plt


from helper_functions import (
    get_and_filter_fns,
    split,
    read_imgs,
    read_bg_imgs,
    get_contours,
)


class Segmenter:
    def __init__(
        self,
        stack_size,
        median_imgs,
        threshold=0,
        min_area=400,
        min_var=30,
        queue_max_size=10,
    ):
        self.stack_size = stack_size
        self.median_imgs = median_imgs
        self.min_area = min_area
        self.threshold = threshold
        self.min_var = min_var

        self.next_bg_queue = Queue(queue_max_size)
        self.current_img_queue = Queue(queue_max_size)

        self.index = 0

    def initial_median_list(self, fns):
        median = np.vstack(
            list(
                map(
                    lambda x: cv.imread(x, cv.IMREAD_GRAYSCALE).flatten(),
                    fns[: self.median_imgs],
                )
            )
        )
        return median

    def get_bg(self, median_list):
        s = time.time()
        median = np.median(median_list, axis=0)
        median = median.reshape((5120, 5120)).astype(np.uint8)

        return median

    def change_bg(self, median_list, new_img):
        median_list[self.index] = new_img
        self.index += 1
        if self.index >= median_list.shape[0]:
            self.index = 0

    def detect_on_corrected_img(self, corrected, img, save_path, fn):
        cnts = get_contours(corrected, self.threshold)
        for cnt in cnts:
            area = cv.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv.boundingRect(cnt)
                if np.var(img[y : y + h, x : x + w]) > self.min_var:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv.imwrite(os.path.join(save_path, fn), img)

    def segment(self, path, save_path):
        fns = get_and_filter_fns(path)
        # fns = fns[:400]
        n_files = len(fns)

        median_list = self.initial_median_list(fns)

        threading.Thread(target=read_imgs, args=(self.current_img_queue, fns)).start()
        threading.Thread(
            target=read_bg_imgs,
            args=(self.next_bg_queue, fns, self.median_imgs, n_files),
        ).start()

        current_bg = self.get_bg(median_list)

        times = []
        mean_img_intensities = []
        mean_bg_intensities = []
        start_total = time.time()
        threads = []
        for i, fn in enumerate(fns):
            s = time.time()
            img = self.current_img_queue.get()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            mean_img_intensities.append(np.mean(gray))  # type: ignore

            if i > self.median_imgs // 2 and i < len(fns) - self.median_imgs // 2:
                next_img = self.next_bg_queue.get().flatten()
                self.change_bg(median_list, next_img)
                current_bg = self.get_bg(median_list)

            mean_bg_intensities.append(np.mean(current_bg))
            corrected = cv.absdiff(gray, current_bg)

            t = threading.Thread(
                target=self.detect_on_corrected_img,
                args=(corrected, img, save_path, str(i) + ".jpg"),
            )

            # while threading.active_count() - 1 >= self.n_threads:
            #     print(threading.active_count())
            #     time.sleep(0.1)
            t.start()

            t = time.time() - s
            times.append(t)
            print(f"{t}s", self.next_bg_queue.qsize())

        print(np.mean(times))
        for t in threads:
            t.join()
        print(
            f"total time: {time.time() - start_total} s, avg per img: {(time.time() - start_total) / n_files}"
        )
        plt.plot(mean_img_intensities, "bo", label="Mean values of images")
        plt.plot(
            mean_bg_intensities,
            label="Mean values of background images",
            color="orange",
        )
        plt.legend()
        plt.savefig(os.path.join(save_path, "development_of_mean_values.png"))


if __name__ == "__main__":
    img_path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
    save_path = "Results/Median"

    # for f in os.listdir(save_path):
    #     os.remove(os.path.join(save_path, f))

    s = Segmenter(stack_size=50, median_imgs=5)
    s.segment(img_path, save_path)
