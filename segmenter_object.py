from importlib.metadata import files
import cv2 as cv
import numpy as np
import os
import time
import threading


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

        self.thread_lock = threading.Lock()

    def check_img(self, img):
        if img is None or img.shape != self.img_shape:
            return False
        return True

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
            if self.check_img(img):
                # if file is usable, add the img to the current background sum
                bg_sum += img
                img_counter += 1
            else:
                # if file is not usable, remove it from local filestack
                index_to_remove = index - start_index
                self.files = (
                    self.files[:index_to_remove] + self.files[index_to_remove + 1 :]
                )

    def preload(self, queue, start_index, end_index):
        index = start_index
        shape = None
        while index <= end_index:
            img = cv.imread(self.files[index], cv.IMREAD_GRAYSCALE)
            if self.check_img(img):
                queue.put(img)
                index += 1
            else:
                index_to_remove = index - start_index
                self.files = (
                    self.files[:index_to_remove] + self.files[index_to_remove + 1 :]
                )
