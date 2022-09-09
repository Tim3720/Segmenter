import cv2 as cv
import numpy as np
import os
import time
import threading
import csv
import queue


class MeanSegmenter:
    def __init__(
        self, master, img_path, save_path, start_index, end_index, bg_size
    ) -> None:
        self.img_path = img_path
        self.save_path = save_path
        self.start_index = start_index
        self.end_index = end_index
        self.bg_size = bg_size
        self.files = master.files[start_index, end_index]
        self.img_shape = master.shape
        self.master = master

    def start_thread(self, func, *args):
        threading.Thread(target=func, args=args).start()

    def compute_starting_bg(self):
        start_index = self.start_index if self.start_index > self.bg_size // 2 else 0
        end_index = start_index + self.bg_size
        end_index = (
            end_index if end_index < self.master.n_files else self.master.n_files - 1
        )

        bg_sum = 0
        img_counter = 0
        for index in range(start_index, end_index + 1):
            filename = self.master.files[index]
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE).astype(np.float64)
            bg_sum += img
            img_counter += 1

        return bg_sum, start_index

    def preload(self, queue, start_index, end_index):
        for index in range(start_index, end_index + 1):
            fn = self.files[index]
            img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
            queue.put((img, fn.split("\\")[-1]))
            index += 1

    def detect(self, corrected_img, current_img, fn):
        thresh = cv.threshold(corrected_img, 0, 255, cv.THRESH_TRIANGLE)[1]
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        counter = 0
        bounding_boxes, filenames = [], []

        for cnt in cnts:
            if cv.contourArea(cnt) < self.master.min_area:
                continue

            x, y, w, h = cv.boundingRect(cnt)
            crop_corrected = corrected_img[y : y + h, x : x + w]
            crop = current_img[y : y + h, x : x + w]

            var = np.var(crop_corrected)
            if var < self.master.min_var:
                continue

            counter += 1
            crop_fn = fn[:-4] + f"_{counter}.jpg"
            bounding_boxes.append((x, y, w, h))
            filenames.append(crop_fn)

            cv.imwrite(os.path.join(self.save_path, crop_fn), crop)

        self.save_info(bounding_boxes, filenames, fn)

        self.master.counter[0] += 1
        p = (self.master.counter[0] / self.master.n_files * 100) // 10 * 10
        if p > self.master.counter[1]:
            self.counter[1] = p  # type: ignore
            print(f"{(self.master.counter[0] * 100 // self.master.n_files)}% done...")

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

    def segment(self):
        start = time.perf_counter()

        bg_sum, start_index = self.compute_starting_bg()
        end_index = self.end_index + self.bg_size // 2
        end_index = (
            self.master.n_files if end_index >= self.master.n_files else end_index
        )
        end_index -= 1

        last_img_queue = queue.Queue(self.master.max_queue_size)
        next_img_queue = queue.Queue(self.master.max_queue_size)
        current_img_queue = queue.Queue(self.master.max_queue_size)

        self.start_thread(
            self.preload, last_img_queue, start_index, end_index - self.bg_size
        )
        self.start_thread(
            self.preload, next_img_queue, start_index + self.bg_size, end_index
        )
        self.start_thread(
            self.preload, current_img_queue, self.start_index, self.end_index
        )

        threads, meta_data = [], []

        for i in range(self.start_index, self.end_index + 1):
            if (
                i > self.bg_size // 2
                and i < self.master.n_files - self.bg_size // 2
                and i != self.start_index
            ):
                last_img, _ = last_img_queue.get(timeout=5)
                next_img, _ = next_img_queue.get(timeout=5)
                bg_sum -= last_img
                bg_sum += next_img
