import os
from statistics import correlation
import cv2 as cv
import numpy as np

img_path = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
files = os.listdir(img_path)

imgs = []
for file in files[:110]:
    img = cv.imread(os.path.join(img_path, file), cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (1000, 1000))
    imgs.append(img)

bg = np.mean(imgs[:100], axis=0).astype(np.uint8)
for img in imgs:
    corrected = cv.absdiff(img, bg)
    neutral_bg = np.ones(bg.shape, np.uint8)  # type: ignore
    neutral_bg = neutral_bg * np.mean(bg)
    img = cv.absdiff(neutral_bg, corrected.astype(np.float64))
    img = img.astype(np.uint8)

    cv.imshow("c", corrected)
    cv.imshow("bg", bg)
    cv.imshow("img", img)

    cv.waitKey()
    cv.destroyAllWindows()
