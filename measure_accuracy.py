import cv2 as cv
import os
import numpy as np
import csv


class Test:
    def __init__(self) -> None:
        pass

    def on_click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            x, y, w, h = self.point_dict[(int(x * 2.56), int(y * 2.56))]
            x = int(x / 2.56)
            y = int(y / 2.56)
            w = int(w / 2.56)
            h = int(h / 2.56)
            cv.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.wrong += 1

    def get_rect(self, x, y):

        left, right, up, down = x, x, y, y
        while self.img[y, left][0] == self.img[y, left][1]:
            left -= 1
        while self.img[y, right][1] == self.img[y, right][0]:
            right += 1

        while self.img[up, x][1] == self.img[up, x][0]:
            up -= 1
        while self.img[down, x][1] == self.img[down, x][0]:
            down += 1

        x = x - left
        y = y - up
        w = left + right
        h = down + up

        return x, y, w, h

    def points_in_rect(self, csv_file):
        with open(csv_file, "r", newline="\n") as f:
            data = list(csv.reader(f, delimiter=";"))
        data = [
            [int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4]] for row in data
        ]
        point_dict = {}
        for x, y, w, h, _ in data:
            for dx in range(w + 1):
                for dy in range(h + 1):
                    new_x = x + dx
                    new_y = y + dy
                    point_dict[(new_x, new_y)] = (x, y, w, h)
        return point_dict, data

    def main(self, path):
        files = os.listdir(path)
        files = [files[2 * i + 1] for i in range(len(files) // 2)]
        fns = list(map(lambda x: os.path.join(path, x), files))

        window_name = "Test"

        cv.namedWindow(window_name)

        cv.setMouseCallback(window_name, self.on_click)

        fns = [os.path.join(path, "20220503-22374880_001.309bar_29.20C_12.jpg")]

        for fn in fns:
            self.img = cv.imread(fn)
            self.img = cv.resize(self.img, (1000, 1000))

            csv_file = fn[:-4] + ".csv"
            self.point_dict, data = self.points_in_rect(csv_file)

            key = 0
            self.wrong = 0
            while key <= 0:
                cv.imshow(window_name, self.img)
                key = cv.waitKey(10)

            print(
                f"Percentage: {round(self.wrong / len(data), 2)}",
                f"Total: {self.wrong}",
            )

            if key == ord("q"):
                break


if __name__ == "__main__":
    path = "Results/Crops_Mean_Half_Resolution"
    t = Test()
    t.main(path)
