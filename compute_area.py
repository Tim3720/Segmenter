import csv
from math import pi, sqrt
import cv2 as cv
import numpy as np
import os


def enhance_contrast(img):
    stds, imgs = [], []
    for alpha in [1.3, 1.35, 1.4, 1.45, 1.5, 1.55]:
        enhanced = img.copy()
        enhanced = cv.bitwise_not(enhanced).astype(np.float64)
        enhanced = enhanced**alpha
        # map values between 0 and 255
        enhanced -= np.max(enhanced) - 255
        enhanced[np.where(enhanced < 0)] = 0
        stds.append(np.std(enhanced))
        imgs.append(enhanced)

    # choose img with best contrast
    best = imgs[stds.index(max(stds))]
    # remove estimated background
    threshold, _ = cv.threshold(best.astype(np.uint8), 0, 255, cv.THRESH_OTSU)
    best -= threshold
    # map values between 0 and 255
    best[np.where(best < 0)] = 0
    best = best / np.max(best) * 255
    best = best.astype(np.uint8)
    return best


def compute_area_from_crop(fn):
    img = cv.imread(fn, cv.IMREAD_GRAYSCALE)

    org_areas = []
    thresh = cv.threshold(cv.bitwise_not(img), 0, 255, cv.THRESH_OTSU)[1]
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        area = cv.contourArea(cnt)
        if area > 25:
            org_areas.append(area)
    if org_areas:
        org_area = max(org_areas)
    else:
        org_area = 0

    areas = []
    enhanced = enhance_contrast(img)
    thresh = cv.threshold(enhanced, 0, 255, cv.THRESH_TRIANGLE)[1]
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        area = cv.contourArea(cnt)
        if area > 25:
            areas.append(area)
    if areas:
        max_area = max(areas)
        if max_area / org_area < 0.5:
            max_area = org_area
    else:
        max_area = org_area

    return max_area, org_area


def compute_area_from_csv(file):
    new_data = []
    path = os.path.split(file)[0]
    with open(file, "r", newline="\n") as f:
        data = list(csv.reader(f, delimiter=";"))
        if not data or len(data[0]) < 5:
            print(f"{file} has wrong information or is empty")
        else:
            for (x, y, w, h, crop_fn) in data:
                cruise, station, device, pos, date, time, id = crop_fn.split("_")
                depth, lat, lon = pos.split("-")
                # time, depth, cruise, id = crop_fn[:-4].split("_")
                # date, station, device, lat, lon = 0, 0, 0, 0, 0
                enhanced_area, area = compute_area_from_crop(
                    os.path.join(path, crop_fn)
                )
                enhanced_esd = 2 * sqrt(enhanced_area / pi)
                esd = 2 * sqrt(area / pi)
                row = [
                    id,
                    crop_fn,
                    x,
                    y,
                    w,
                    h,
                    area,
                    esd,
                    enhanced_area,
                    enhanced_esd,
                    depth,
                    date,
                    time,
                    cruise,
                    station,
                    device,
                    lat,
                    lon,
                    date,
                    time,
                ]
                new_data.append(row)

    if new_data:
        with open(file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            header = [
                "ID",
                "filename",
                "min_x in px",
                "min_y in px",
                "width in px",
                "height in px",
                "area in px",
                "esd in px",
                "enhanced_area in px",
                "enhanced_esd in px",
                "depth in m",
                "date",
                "time",
                "cruise",
                "station",
                "device",
                "latitude",
                "longitude",
                "date",
                "time",
            ]
            writer.writerow(header)
            for row in new_data:
                writer.writerow(row)

    return new_data


if __name__ == "__main__":
    from multiprocessing import Pool
    from time import perf_counter

    path = "Results/Crops_Mean_Half_Resolution"
    files = os.listdir(path)
    csv_files = [os.path.join(path, file) for file in files if file.endswith(".csv")]
    s = perf_counter()
    with Pool() as pool:
        res = pool.map(compute_area_from_csv, csv_files)

    # data = []
    # for res_file in res:
    #     data.extend(res_file)
    # with open(os.path.join(path, "object_information.csv"), "w") as f:
    #     writer = csv.writer(f, delimiter=";")
    # header = [
    #     "ID",
    #     "filename",
    #     "min_x",
    #     "min_y",
    #     "width",
    #     "height",
    #     "area",
    #     "esd",
    #     "enhanced_area",
    #     "enhanced_esd",
    #     "depth",
    #     "date",
    #     "time",
    #     "cruise",
    #     "station",
    #     "device",
    #     "latitude",
    #     "longitude",
    #     "date",
    #     "time",
    # ]
    #     writer.writerow(header)
    #     for row in data:
    #         writer.writerow(row)

    duration = perf_counter() - s
    print(
        f"Total duration: {duration:.2f}s, Avg time per file: {duration / len(csv_files):.4f}s"
    )
