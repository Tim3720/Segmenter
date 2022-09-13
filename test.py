import os, time
import cv2 as cv
import numpy as np


def enhance_contrast(img):
    stds, imgs = [], []
    for alpha in [1.3, 1.35, 1.4, 1.45, 1.5, 1.55]:
        enhanced = img.copy()
        enhanced = cv.bitwise_not(enhanced).astype(np.float64)
        enhanced = enhanced**alpha
        # set max value to 255
        enhanced -= np.max(enhanced) - 255
        enhanced[np.where(enhanced < 0)] = 0
        stds.append(np.std(enhanced))
        imgs.append(enhanced)

    best = imgs[stds.index(max(stds))]
    threshold, _ = cv.threshold(best.astype(np.uint8), 0, 255, cv.THRESH_OTSU)
    best -= threshold
    best[np.where(best < 0)] = 0
    best = best / np.max(best) * 255
    best = best.astype(np.uint8)
    return best


save_path = "Results/Crops_Mean_Half_Resolution"


def get_area(file):
    s = time.perf_counter()
    img = cv.imread(os.path.join(save_path, file), cv.IMREAD_GRAYSCALE)
    enhanced = enhance_contrast(img)

    max_areas = []
    max_cnts = []
    threshs = [
        cv.threshold(cv.bitwise_not(img), 0, 255, cv.THRESH_OTSU)[1],
        cv.threshold(enhanced, 0, 255, cv.THRESH_TRIANGLE)[1],
    ]

    for i, thresh in enumerate(threshs):
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        areas = []

        for cnt in cnts:
            areas.append(cv.contourArea(cnt))

        max_cnts.append(cnts[areas.index(max(areas))])
        max_areas.append(max(areas))
    area = max_areas[1] if max_areas[1] / max_areas[0] > 0.3 else max_areas[0]
    cnt = max_cnts[1] if max_areas[1] / max_areas[0] > 0.3 else max_cnts[0]
    cv.drawContours(img, [cnt], -1, 255, 1)
    duration = time.perf_counter() - s
    return area, duration


if __name__ == "__main__":
    import random, time
    import matplotlib.pyplot as plt
    from multiprocessing import Pool

    areas, times = [], []
    files = os.listdir(save_path)
    files = [file for file in files if file.endswith(".jpg")]

    s = time.perf_counter()
    with Pool() as pool:

        res = pool.map(get_area, files)

        # c = np.concatenate(
        #     [img, enhanced],
        #     axis=1,
        # )
        # c = np.concatenate(
        #     [
        #         img,
        #         enhanced,
        #         cv.threshold(cv.bitwise_not(img), 0, 255, cv.THRESH_OTSU)[1],
        #         cv.threshold(enhanced, 0, 255, cv.THRESH_OTSU)[1],
        #         cv.threshold(enhanced, 0, 255, cv.THRESH_TRIANGLE)[1],
        #     ],
        #     axis=1,
        # )
        # cv.imshow("C", c)
        # key = cv.waitKey()
        # if key == ord("q"):
        #     break
        # cv.destroyAllWindows()
    times = [elem[1] for elem in res]
    areas = [elem[0] for elem in res]

    duration = time.perf_counter() - s
    print(f"Avg time: {duration / len(files):.4f}s, Total: {duration:.2f}s", max(areas))
    plt.hist(areas, bins=range(0, 80000, 2500))
    plt.show()
