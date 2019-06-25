from numpy import *
import math
import numpy as np


def lti(data_train, sigma=0.6, sigma_kernel=0.3, kernel_type='lin', window=2):
    window_threshold = 0  # window terpenuhi
    head_cut = 0  # index awal untuk dipotong
    cut = False
    while not cut:
        if (data_train[head_cut] != 0):
            window_threshold += 1
        else:
            window_threshold = 0
        if (window_threshold == window):
            cut = True
        head_cut += 1

    if (head_cut == window):
        hasil = proses(data_train, sigma, sigma_kernel, kernel_type, window)
    else:
        hasil = proses(data_train[head_cut:len(data_train)], sigma, sigma_kernel, kernel_type, window)
        data_proses = np.append(data_train[0:(head_cut - 1)])
        data_proses = np.flip(data_proses, 0)
        hasil = proses(data_proses, sigma, sigma_kernel, kernel_type, window)

        hasil = np.flip(hasil, 0)
    return hasil


def proses(data_train, sigma, sigma_kernel, kernel_type, window):
    pattern = create_pattern(data_train, window)
    C = sigma
    k1 = sigma_kernel  # sigma untuk kernel
    kernel = kernel_type
    kTup = (kernel, k1)
    ##2.训练模型
    # print('----------------------------2.Train Model------------------------------')
    alphas, b, K = leastSquares(pattern[0], pattern[1], C, kTup)
    ##3.计算训练误差
    # print('----------------------------3.Calculate Train Error--------------------')
    error = 0.0
    test = []
    for i in range(0, len(pattern[2])):
        result = predict(alphas, b, pattern[0], pattern[2][i], kTup)
        test = np.append(test, result)
        # error += abs(result - test_result[i])
    # abs_error = error / len(pattern[2])
    final = insert_data(data_train, test)
    return final


def insert_data(datain, data_imputasi):
    i = 0
    if(len(data_imputasi)!=0):
        for j in range(0, len(datain)):
            if (datain[j] == 0):
                datain[j] = data_imputasi[i]
                i += 1
    return datain


def create_pattern(datain, window=2):
    if window < 2:
        window = 2

    # jika data pertama adalah null
    while (datain[0] == 0):
        datain = datain[1:len(datain)]

    n_non_na = 0
    n_na = 0
    index_non_na = []
    index_na = []
    for i in range(0, len(datain)):
        if (datain[i] == 0):
            n_na = n_na + 1
            index_na = np.append(index_na, i)
        else:
            index_non_na = np.append(index_non_na, i)
            n_non_na = n_non_na + 1

    # training_pattern : x
    training_pattern = mat(zeros(((n_non_na - window), (2 * window + 1))))
    # target_pattern : y dari training_pattern
    target_pattern = []

    # test_pattern : data yang akan diimputasi
    test_pattern = mat(zeros(((n_na), (2 * window + 1))))

    # Membuat Training pattern
    for i in range(0, shape(training_pattern)[0]):
        pattern = mat(zeros((2 * window) + 1)).T

        for j in range(0, window):
            pattern[j + window] = index_non_na[i + j]
            pattern[j] = datain[pattern[j + window].astype(int)]

        pattern[window * 2] = index_non_na[i + window]
        target_pattern = np.append(target_pattern, datain[index_non_na[i + window].astype(int)])

        # Jika pattern pertama tidak nol
        if (pattern[window] != 0):
            difference = pattern[window].astype(int)
            for j in range(window, shape(pattern)[0]):
                pattern[j] = pattern[j] - difference

        training_pattern[i,] = pattern.T

    # Membuat Test Pattern
    for i in range(0, shape(test_pattern)[0]):
        pattern = mat(zeros((2 * window + 1))).T

        index_non_na_before = np.where(index_non_na < index_na[i])[0]
        for j in range(0, window):
            pattern[j + window] = index_non_na[index_non_na_before[len(index_non_na_before) - window - 1 + j]]
            pattern[j] = datain[pattern[j + window].astype(int)]

        pattern[window * 2] = index_non_na[index_non_na_before[len(index_non_na_before) - 1]]

        # Jika pattern pertama tidak nol
        if (pattern[window] != 0):
            difference = pattern[window].astype(int)
            for j in range(window, shape(pattern)[0]):
                pattern[j] = pattern[j] - difference
        test_pattern[i,] = pattern.T

    return training_pattern, target_pattern, test_pattern


def kernelTrans(X, A, kTup):
    X = mat(X)  # matrix data X
    m, n = shape(X)  # get dimension X, m = baris, n = kolom
    K = mat(zeros((m, 1)))  # membuat matrik kosong dengan kolom 1 dan baris m
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, kTup):
        self.X = dataMatIn  # data matrix X
        self.labelMat = classLabels  # target matrix
        self.C = C
        self.m = shape(dataMatIn)[0]  # length data
        self.alphas = mat(zeros((self.m, 1)))  # matrix aplha dengan isi null
        self.b = 0
        self.K = mat(zeros((self.m, self.m)))  # matrix k ukuran mxm berisi nol
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def leastSquares(dataMatIn, classLabels, C, kTup):
    '''最小二乘法求解alpha序列
     C = sigma
    '''
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, kTup)
    ##1.参数设置
    unit = mat(ones((oS.m, 1)))  # [1,1,...,1].T
    I = eye(oS.m)  # matriks idenditas
    zero = mat(zeros((1, 1)))
    upmat = hstack((zero, unit.T))
    downmat = hstack((unit, oS.K + I / float(C)))
    ##2.方程求解
    completemat = vstack((upmat, downmat))  # lssvm中求解方程的左边矩阵
    rightmat = vstack((zero, oS.labelMat))  # lssvm中求解方程的右边矩阵
    b_alpha = completemat.I * rightmat
    oS.b = b_alpha[0, 0]
    for i in range(oS.m):
        oS.alphas[i, 0] = b_alpha[i + 1, 0]
    return oS.alphas, oS.b, oS.K


def predict(alphas, b, dataMat, testVec, kTup):
    Kx = kernelTrans(dataMat, testVec, kTup)
    predict_value = Kx.T * alphas + b

    return float(predict_value)



def Kernel(x, y, sigma, gamma):
    delta = x - y
    sumSqure = float(delta.dot(delta.T))
    result = math.exp(-0.5 * sumSqure / (sigma ** 2))
    return result


def Train(x, y, sigma, gamma):
    length = len(x) + 1

    A = np.zeros(shape=(length, length))
    A[0][0] = 0
    A[0, 1:length] = y.T
    A[1:length, 0] = y
    A[1:length, 1:length] = Omega(x, y, sigma) + np.eye(length - 1) / gamma

    B = np.ones(length, 1)
    B[0][0] = 0

    return np.linalg.solve(A, B)


def Omega(x, y, sigma):
    length = len(x)
    omega = np.zeros(shape=(length, length))
    for i in range(length):
        for j in range(length):
            omega[i, j] = y[i] * y[j] * Kernel(x[i], x[j], sigma)
    return omega