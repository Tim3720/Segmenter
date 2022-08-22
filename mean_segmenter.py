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
        n_threads,
        threshold=0,
        min_area=400,
        min_var=30,
        queue_max_size=10,
        save_full_img=False,
        master=None,
    ):
        self.stack_size = stack_size
        self.n_threads = n_threads
        self.min_area = min_area
        self.threshold = threshold
        self.min_var = min_var
        self.save_full_img = save_full_img
        self.master = master

        self.next_img_queue = Queue(queue_max_size)
        self.last_img_queue = Queue(queue_max_size)
        self.current_img_queue = Queue(queue_max_size)

        self.running = False
        self.lock = threading.Lock()
        self.total_counter = 0

    def compute_initial_stack(self, fns):
        def sum_in_threads(fns, sum_list):
            sum = 0
            for fn in fns:
                img = cv.imread(fn, cv.IMREAD_GRAYSCALE).astype(np.float64)
                sum += img
            sum_list.append(sum)

        sum_list = []
        stack_files = fns[: self.stack_size]
        stack_files = split(stack_files, self.stack_size // self.n_threads)
        threads = []
        for sample in stack_files:
            t = threading.Thread(target=sum_in_threads, args=(sample, sum_list))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return np.sum(sum_list, axis=0)

    def detect_on_corrected_img(self, corrected, img, save_path, fn):
        cnts = get_contours(corrected, self.threshold)
        counter = 0
        for i, cnt in enumerate(cnts):
            area = cv.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv.boundingRect(cnt)
                crop = img[y : y + h, x : x + w]
                if np.var(crop) > self.min_var:
                    counter += 1
                    if self.save_full_img:
                        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                        # cv.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                        cv.putText(
                            img,
                            str(round(np.var(crop))),
                            (x, y),
                            cv.FONT_HERSHEY_DUPLEX,
                            1,
                            (0, 255, 0),
                            1,
                        )
                    else:
                        cv.imwrite(
                            os.path.join(save_path, fn + "_" + str(counter) + ".jpg"),
                            crop,
                        )

            # if not self.running:
            #     break

        with self.lock:
            self.total_counter += counter

        if self.save_full_img:
            cv.imwrite(os.path.join(save_path, fn + ".jpg"), img)

    def segment(self, path, save_path):
        self.running = True
        fns = get_and_filter_fns(path)
        # fns = fns[:100]
        n_files = len(fns)
        if self.master != None:
            self.master.info["n_files"] = n_files

        current_bg_sum = self.compute_initial_stack(fns)

        t1 = threading.Thread(
            target=read_imgs, args=(self.current_img_queue, fns, self)
        )
        t1.start()

        t2 = threading.Thread(
            target=read_bg_imgs,
            args=(self.last_img_queue, fns, 0, n_files - self.stack_size, self),
        )
        t2.start()

        t3 = threading.Thread(
            target=read_bg_imgs,
            args=(self.next_img_queue, fns, self.stack_size, n_files, self),
        )
        t3.start()

        times = []
        mean_img_intensities = []
        mean_bg_intensities = []
        start_total = time.time()
        threads = [t1, t2, t3]

        if self.master != None:
            self.master.info["total_time"] = 0

        for i, fn in enumerate(fns):
            if not self.running:
                break
            s = time.time()
            img = self.current_img_queue.get(timeout=2)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            mean_img_intensities.append(np.mean(gray))  # type: ignore

            if i > self.stack_size // 2 and i < n_files - self.stack_size // 2:
                next_img = self.next_img_queue.get(timeout=2)
                last_img = self.last_img_queue.get(timeout=2)
                current_bg_sum -= last_img
                current_bg_sum += next_img

            current_bg = (current_bg_sum / self.stack_size).astype(np.uint8)
            mean_bg_intensities.append(np.mean(current_bg))
            corrected = cv.absdiff(gray, current_bg)
            thread = threading.Thread(
                target=self.detect_on_corrected_img,
                args=(corrected, img, save_path, fn.split("\\")[-1][:-4]),
            )

            while threading.active_count() - 1 >= self.n_threads:
                print(threading.active_count())
                print(list(threading.enumerate()))
                time.sleep(0.1)
            thread.start()

            t = time.time() - s
            times.append(t)

            if self.master != None:
                self.running = self.master.running

                self.master.info["current_file"] = fn.split("\\")[-1]
                self.master.info["time"] = round(t, 2)
                self.master.info["segmented_files"] = i + 1
                self.master.info["total_time"] = round(
                    t + self.master.info["total_time"], 2
                )
                self.master.info["avg_time"] = round(
                    self.master.info["total_time"] / (i + 1), 2
                )
                self.master.info["object_counter"] = self.total_counter

        for t in threads:
            t.join()

        print("Segmenter", list(threading.enumerate()))

        plt.plot(mean_img_intensities, "bo", label="Mean values of images")
        plt.plot(
            mean_bg_intensities,
            label="Mean values of background images",
            color="orange",
        )
        plt.legend()
        plt.savefig(os.path.join(save_path, "development_of_mean_values.png"))
        plt.close()


if __name__ == "__main__":
    img_path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
    save_path = "Results/Stack"

    for f in os.listdir(save_path):
        os.remove(os.path.join(save_path, f))

    s = Segmenter(stack_size=100, n_threads=8, save_full_img=False)
    s.segment(img_path, save_path)
