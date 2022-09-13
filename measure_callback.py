import csv, os


def read_csv(file, delimiter):
    with open(file, "r", newline="") as f:
        data = csv.reader(f, delimiter=delimiter)
        print(list(data)[0])
    with open("test.csv", "w", newline="") as f:
        data = csv.reader(f, delimiter=delimiter)
        print(list(data)[0])
        
read_csv("M181-fin7.csv", ",")

