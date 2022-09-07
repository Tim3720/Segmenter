import os
import matplotlib.pyplot as plt
import csv

mean, std = [], []
with open("Results/Crops_Mean_Half_Resolution/1_meta_data.csv", newline="\n") as f:
    data = list(csv.reader(f, delimiter=";"))
    for row in data:
        mean.append(float(row[1]))
        std.append(float(row[2]) * 5)

plt.plot(mean, "bo")
# plt.plot(std, "go")
plt.show()
