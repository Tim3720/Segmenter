import os
import cv2 as cv
import numpy as np
import time
from queue import Queue


def get_and_filter_fns(path, valid_endings=["jpg", "png", "tif"]):
    fns = os.listdir(path)
    filtered_fns = []
    for fn in fns:
        if fn[-3:] in valid_endings:
            filtered_fns.append(os.path.join(path, fn))

    return filtered_fns


def split(list, intervall_len):
    elements = len(list) // intervall_len
    if len(list) % intervall_len != 0:
        return [
            list[i * intervall_len : (i + 1) * intervall_len] for i in range(elements)
        ] + [list[elements * intervall_len :]]
    else:
        return [
            list[i * intervall_len : (i + 1) * intervall_len] for i in range(elements)
        ]


def read_bg_imgs(queue: Queue, fns, start_index, max_index, master):
    i = start_index
    while i < max_index and master.running:
        fn = fns[i]
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE).astype(np.float64)
        try:
            queue.put_nowait(img)
            i += 1
        except:
            time.sleep(0.1)


def read_imgs(queue: Queue, fns, master):
    i = 0
    while i < len(fns) and master.running:
        img = cv.imread(fns[i])
        try:
            queue.put_nowait(img)
            i += 1
        except:
            time.sleep(0.1)


def get_contours(img, threshold):
    if type(threshold) != int:
        raise TypeError("threshold must be int")

    if threshold == 0:
        method = cv.THRESH_TRIANGLE
    elif threshold == -1:
        method = cv.THRESH_OTSU
    else:
        method = cv.THRESH_BINARY

    thresh = cv.threshold(img, threshold, 255, method)[1]
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    return cnts
