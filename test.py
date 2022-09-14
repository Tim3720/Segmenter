import os
import csv

path = "Results/Crops_Mean_Half_Resolution"
files = os.listdir(path)
csv_files = [
    os.path.join(path, file)
    for file in files
    if file.endswith(".csv") and file != "meta_data.csv"
]

counter = 0
for file in csv_files:
    with open(file, "r") as f:
        data = list(csv.reader(f, delimiter=","))
        for row in data[1:]:
            if float(row[6]) < 25 or float(row[10]) < 25:
                print("Area is small", row[1], row[6], row[10])
                counter += 1

print(counter)
