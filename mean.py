import numpy as np


def my_mean(datain):
    print(datain)
    data_not_na = []

    for i in range(0, len(datain)):
        if (datain[i] != 0):
            data_not_na = np.append(data_not_na, datain[i])

    mean_val = np.mean(data_not_na)

    for i in range(0, len(datain)):
        if (datain[i] == 0):
            datain[i] = mean_val
    print(datain)
    return datain
