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
