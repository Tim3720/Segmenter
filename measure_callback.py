import csv, os


def read_csv(file, delimiter=","):
    with open(file, "r", newline="\n") as f:
        data = list(csv.reader(f, delimiter=delimiter))
    return data


def compute_point_dict(csv_data):
    point_dict = {}
    for obj in csv_data:
        x, y, w, h, crop_fn = obj
        for dx in range(w):
            for dy in range(h):
                new_x = x + dx
                new_y = y + dy
                point_dict[(new_x, new_y)] = crop_fn
    return point_dict


def get_img_dict(labeled_data):
    img_dict = {}
    for row in labeled_data:
        fn, _, _, _, x_min, y_min, x_max, y_max = row
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        if not fn in img_dict.keys():
            img_dict[fn] = []
        c_x = round((x_min + x_max) / 2)
        c_y = round((y_min + y_max) / 2)
        img_dict[fn].append((c_x, c_y))
    return img_dict


def measure_callback(path_to_labeled, path_to_crops):
    labeled_data = read_csv(path_to_labeled)[1:]
    img_dict = get_img_dict(labeled_data)
    for img in img_dict.keys():
        crop_csv = os.path.join(path_to_crops, img[:-4] + ".csv")
        crop_csv_data = read_csv(crop_csv, delimiter=";")
        point_dict = compute_point_dict(crop_csv_data)
        for point in img_dict[img]:
            if point in point_dict.keys():
                print(point_dict[point])
            else:
                print("Not found", img)


if __name__ == "__main__":
    path = "Results/Crops_Mean_Half_Resolution"
    files = os.listdir(path)
    # csv_files = [os.path.join(path, file) for file in files if file.endswith(".csv")]
    # data = read_csv(csv_files[0], delimiter=";")
    # point_dict = compute_point_dict(data)
    measure_callback("M181-fin7.csv", 0)
