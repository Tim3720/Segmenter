import csv, os


def read_csv(file, delimiter):
    with open(file, "r", newline="") as f:
        data = csv.reader(f, delimiter=delimiter)
        print(list(data)[0])


read_csv("M181-fin7.csv", ",")


def point_dict(csv_data):
    point_dict = {}
    for obj in csv_data:
        x, y, w, h, crop_fn = obj
        for dx in range(w):
            for dy in range(h):
                new_x = x + dx
                new_y = y + dy
                point_dict[(new_x, new_y)] = crop_fn
    return point_dict
