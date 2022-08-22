import cv2 as cv
import numpy as np
import time
import os


def compute_median(imgs, n_threads):
    img_shape = imgs[0].shape
    img = np.dstack(imgs)


img_path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
fns = os.listdir(img_path)
fns = list(map(lambda x: os.path.join(img_path, x), fns))

start = time.perf_counter()
bg_list = []
for i in range(5):
    img = cv.imread(fns[i], cv.IMREAD_GRAYSCALE)
    bg_list.append(img)
bg = np.median(bg_list, axis=0).astype(np.uint8)

end = time.perf_counter()
duration = end - start
print(f"Read in: {duration: .2f}s")


start = time.perf_counter()
img = cv.imread(fns[5], cv.IMREAD_GRAYSCALE)
bg_list[0] = img
bg2 = np.median(bg_list, axis=0).astype(np.uint8)
end = time.perf_counter()
duration = end - start
print(f"Read in: {duration: .2f}s")

# cv.imshow("bg1", cv.resize(bg, (1000, 1000)))
# cv.imshow("bg2", cv.resize(bg2, (1000, 1000)))
# cv.waitKey()
