from numpy import *
import math
from scipy.interpolate import interp1d
import warnings
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def hotdeck(data):
	reverse = False
	for i in range(0, len(data)):
		if (data[i] == 0):
			if (i != 0):
				data[i] = data[i - 1]
			else:
				reverse = True
	if reverse:
		data = flip(data, 0)
		for i in range(1, len(data)):
			if (data[i] == 0):
				data[i] = data[i - 1]
		data = flip(data, 0)
	return data


def mymean(datain):
	data_not_na = []

	for i in range(0, len(datain)):
		if (datain[i] != 0):
			data_not_na = append(data_not_na, datain[i])

	mean_val = mean(data_not_na)

	for i in range(0, len(datain)):
		if (datain[i] == 0):
			datain[i] = mean_val

	return datain


def interpolasi(data):
	i = 1
	while (data[0] == 0):
		if (data[i] != 0):
			j = copy.deepcopy(i)
			while (j != 0):
				data[j - 1] = data[j]
				j -= 1
		else:
			i += 1

	i = 1
	while (data[len(data) - 1] == 0):
		if (data[[len(data) - 1 - i]] != 0):
			j = copy.deepcopy(i)
			j = len(data) - 1 - j
			while (j != (len(data) - 1)):
				data[j + 1] = data[j]
				j += 1
		else:
			i += 1
	# indeks mulai dari 0
	indices = array(range(0, len(data)))
	indices_non_na = []
	y = []
	for i in range(0, len(data)):
		if (data[i] != 0):
			indices_non_na = append(indices_non_na, i)
			y = append(y, data[i])

	f = interp1d(indices_non_na, y)
	return f(indices)


# ==============================================PSF=================================================#
# =============================================START================================================#
def psf(data, n_ahead, k=list(range(2, 11)), w=list(range(1, 11)), cycle=4):
	length = len(data)

	# Memotong panjang data apabila tidak kelipatan cycle
	fit = length % cycle
	if (fit > 0):
		#         warnings.warn("Panjang series harus kelipatan "+str(cycle))
		data = data[0:(length - fit)]
	length = length - fit

	# Memotong panjang prediksi apabila tidak kelipatan cycle
	original_n_ahead = n_ahead
	fit = n_ahead % cycle
	if (fit > 0):
		n_head = cycle * math.ceil(n_ahead / cycle)
	#         warnings.warn("Panjang prediksi harus kelipatan "+str(cycle))

	# Melakukan normalisasi
	dmin = min(data)
	dmax = max(data)
	for i in range(0, length):
		data[i] = (data[i] - dmin) / (dmax - dmin)

	# Memasukkan data ke dalam variable sesuai cycle
	series = zeros(shape=(int(length / cycle), cycle))
	row = 0
	for i in range(0, length):
		if (i != 0 and i % cycle == 0):
			row = row + 1
		series[row, i % cycle] = data[i]

	# Mencari K maksimum
	if (len(k) > 1):
		k = optimum_k(series, k)

	if (len(w) > 1):
		w = optimum_w(series, k, w, cycle)
	pred = psf_predict(series, k, w, n_ahead, cycle)[0:original_n_ahead]

	for i in range(0, len(pred)):
		pred[i] = pred[i] * (dmax - dmin) + dmin

	return [pred, k, w]


def optimum_k(data, k=list(range(2, 11))):
	best_k = 1
	best_s = -1 * math.inf
	n = len(data)
	for i in k:
		if (i > 1 and i < n):
			labels = KMeans(n_clusters=i, random_state=0).fit(data).labels_
			s = silhouette_score(data, labels)
			if (s > best_s):
				best_s = s
				best_k = i
	return best_k


def psf_predict(dataset, k, w, n_ahead, cycle):
	temp = dataset
	n_ahead_cycle = n_ahead / cycle  # banyaknya set head cycle

	iterate = 1;
	cw = w
	n = len(dataset)
	while (iterate <= n_ahead_cycle):
		if (cw == w):
			clusters = KMeans(n_clusters=k).fit(temp).labels_

		pattern = clusters[(n - cw):n]

		neighbors = []
		for i in range(0, (len(clusters) - cw)):
			if ((clusters[i:(i + cw)] == pattern).any()):
				neighbors = append(neighbors, i)

		if (len(neighbors) == 0):
			#             cw = cw -1
			#             if(cw==0):
			row = dataset.shape[0]
			column = dataset.shape[1]
			new_temp = resize(temp, (row + 1, column))
			new_temp[row] = dataset[len(dataset) - 1]
			temp = new_temp
			cw = w;
			iterate = iterate + 1
		else:
			for i in range(0, len(neighbors)):
				neighbors[i] = neighbors[i] + cw

			pred = sum(temp[neighbors.astype(int)]) / len(neighbors)
			row = dataset.shape[0]
			column = dataset.shape[1]
			new_temp = resize(temp, (row + 1, column))
			new_temp[row] = pred
			temp = new_temp
			cw = w;
			iterate = iterate + 1

	result = temp[int(len(temp) - n_ahead_cycle):int(len(temp))]
	result_array = []
	for i in range(0, result.shape[0]):
		for j in range(0, result.shape[1]):
			result_array = append(result_array, result[i, j])

	return result_array


def optimum_w(dataset, k, w_values, cycle):
	test = dataset[len(dataset) - 1]
	training = dataset[0:(len(dataset) - 1)]
	n = len(training)
	best_w = 0
	min_err = math.inf
	for w in w_values:
		if (w > 0 and w < n):
			pred = psf_predict(training, k, w, cycle, cycle)

			err = sum(abs(pred - test)) / cycle

			if ((err < min_err).any()):
				min_err = err
				best_w = w
	return best_w


def missing_patch(dataIn):
	length = len(dataIn)

	get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
	list_na = get_indexes(0, dataIn)
	m = abs(diff(list_na))  # Menghasilkan selisih tiap index NA
	m = append(m, 100)  # Menambah angka 100 di belakang m
	x = list_na[0]  # get first index of NA
	y = []

	for i in range(0, len(list_na)):
		if (m[i] != 1):  # jika selisih tidak sama dengan 1
			if (i == len(list_na) - 1):
				x = append(x, 0)
			else:
				x = append(x, list_na[i + 1])
			y = append(y, list_na[i])
	x = x[0:len(x) - 1]
	z = []
	for i in range(0, len(x)):
		z = append(z, (y[i] - x[i] + 1).astype(int))

	data0 = column_stack((x, y, z))
	return data0


def heal_data(data, patch):
	datain = copy.deepcopy(data)
	patch1 = patch
	length = len(datain)
	data_wt_na = []
	# Membuang data NA
	for i in range(0, len(datain)):
		if (datain[i] != 0):
			data_wt_na = append(data_wt_na, datain[i])
	lens = len(data_wt_na)
	while (patch.shape[0] != 0):
		max_z_patch_index = 0
		for i in range(0, patch.shape[0]):
			if (max_z_patch_index < patch[i, 2]):
				max_z_patch_index = i
		f_first = int(patch[max_z_patch_index, 0])
		f_last = int(patch[max_z_patch_index, 1])
		n_ahead = int(patch[max_z_patch_index, 2])
		n_wt_na = len(data_wt_na)
		if (f_first < 0.2 * n_wt_na):
			dataset = []
			for i in range((f_last + 1), len(datain)):
				if (datain[i] != 0):
					dataset = append(dataset, datain[i])
			dataset = flip(dataset, 0)
			x2 = psf(dataset, n_ahead=n_ahead, cycle=4)
			x2 = x2[0]
			x2 = flip(x2, 0)
			x3 = x2
		if ((f_first >= 0.2 * lens) and (f_last <= 0.8 * lens)):
			# dataset = data_wt_na
			dataset = []
			for i in range(0, f_first):
				if (datain[i] != 0):
					dataset = append(dataset, datain[i])
			x1 = psf(dataset, n_ahead=n_ahead, cycle=4)
			x1 = x1[0]

			dataset = []
			for i in range((f_last + 1), len(datain)):
				if (datain[i] != 0):
					dataset = append(dataset, datain[i])
			dataset = flip(dataset, 0)
			# f_first_rev = n_wt_na - f_last
			x2 = psf(dataset, n_ahead=n_ahead, cycle=4)
			x2 = x2[0]
			x2 = flip(x2, 0)
			x3 = []
			for i in range(0, len(x2)):
				x3 = append(x3, ((x1[i] + x2[i]) / 2))

		if (f_last > 0.8 * lens):
			dataset = []
			for i in range(0, f_first):
				if (datain[i] != 0):
					dataset = append(dataset, datain[i])
			# dataset = data_wt_na
			x3 = psf(dataset, n_ahead=n_ahead, cycle=4)
			x3 = x3[0]
		datain = insert_patch(datain, f_first, x3)
		patch = delete(patch, (max_z_patch_index), axis=0)

	return datain


def insert_patch(datain, pos, dataInsert):
	for i in range(0, len(dataInsert)):
		datain[pos + i] = dataInsert[i]
	return datain


# ==============================================PSF=================================================#
# ==============================================END=================================================#

# ==============================================LTI=================================================#
# =============================================START================================================#
def lti(data_train, sigma=0.6, sigma_kernel=0.3, kernel_type='lin', window=2):
	# Membuat data di awal sehingga sampai data sejumlah window tidak missing
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
		if (head_cut == (len(data_train) - 1)):
			cut = True

	if (head_cut == window):
		hasil = proses(data_train, sigma, sigma_kernel, kernel_type, window)
	else:
		if (head_cut > (0.5 * len(data_train))):
			warnings.warn("Window terlalu lebar")
			return
		hasil = proses(data_train[(head_cut - window):len(data_train)], sigma, sigma_kernel, kernel_type, window)

		data_proses = []
		data_proses = append(data_proses, data_train[0:(head_cut - window)])
		data_proses = append(data_proses, hasil)

		data_proses = flip(data_proses, 0)
		hasil = proses(data_proses, sigma, sigma_kernel, kernel_type, window)

		hasil = flip(hasil, 0)
	return hasil


def proses(data_train, sigma, sigma_kernel, kernel_type, window):
	pattern = create_pattern(data_train, window)
	C = sigma
	k1 = sigma_kernel  # sigma untuk kernel
	kernel = kernel_type
	kTup = (kernel, k1)
	alphas, b, K = leastSquares(pattern[0], pattern[1], C, kTup)
	error = 0.0
	test = []
	for i in range(0, len(pattern[2])):
		result = predict(alphas, b, pattern[0], pattern[2][i], kTup)
		test = append(test, result)
	final = insert_data(data_train, test)
	return final


def insert_data(datain, data_imputasi):
	i = 0
	if (len(data_imputasi) != 0):
		for j in range(0, len(datain)):
			if (datain[j] == 0):
				datain[j] = copy.deepcopy(data_imputasi[i])
				i += 1
	return datain


def create_pattern(datain, window=2):
	if window < 2:
		window = 2

	n_non_na = 0
	n_na = 0
	index_non_na = []
	index_na = []
	for i in range(0, len(datain)):
		# Mengumpulkan indeks yang na
		if (datain[i] == 0):
			n_na = n_na + 1
			index_na = append(index_na, i)
		# Mengumpulkan indeks yang Non na
		else:
			index_non_na = append(index_non_na, i)
			n_non_na = n_non_na + 1

	# Membuat training matriks X
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
		target_pattern = append(target_pattern, datain[index_non_na[i + window].astype(int)])

		# Jika pattern pertama tidak nol
		if (pattern[window] != 0):
			difference = copy.deepcopy(pattern[window]).astype(int)
			for j in range(window, shape(pattern)[0]):
				pattern[j] = pattern[j] - difference

		training_pattern[i,] = pattern.T

	# Membuat Test Pattern
	for i in range(0, shape(test_pattern)[0]):
		pattern = mat(zeros((2 * window + 1))).T

		index_non_na_before = where(index_non_na < index_na[i])[0]
		for j in range(0, window):
			pattern[j + window] = index_non_na[index_non_na_before[len(index_non_na_before) - window - 1 + j]]
			index = copy.deepcopy(pattern[j + window])
			pattern[j] = datain[index.astype(int)]

		pattern[window * 2] = index_non_na[index_non_na_before[len(index_non_na_before) - 1]]

		# Jika pattern pertama tidak nol
		if (pattern[window] != 0):
			difference = copy.deepcopy(pattern[window]).astype(int)
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
	elif kTup[0] == 'polinom':
		K = (X * A.T + 1)
		for i in range(0, len(K)):
			K[i] = K[i] ** kTup[1]
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
	oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, kTup)

	unit = mat(ones((oS.m, 1)))  # membuat matriks berisi angka satu dgn row sebanyak data
	I = eye(oS.m)  # matriks idenditas
	zero = mat(zeros((1, 1)))
	upmat = hstack((zero, unit.T))
	downmat = hstack((unit, oS.K + I / float(C)))
	completemat = vstack((upmat, downmat))
	rightmat = vstack((zero, oS.labelMat))
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

	A = zeros(shape=(length, length))
	A[0][0] = 0
	A[0, 1:length] = y.T
	A[1:length, 0] = y
	A[1:length, 1:length] = Omega(x, y, sigma) + eye(length - 1) / gamma

	B = ones(length, 1)
	B[0][0] = 0

	return linalg.solve(A, B)


def Omega(x, y, sigma):
	length = len(x)
	omega = zeros(shape=(length, length))
	for i in range(length):
		for j in range(length):
			omega[i, j] = y[i] * y[j] * Kernel(x[i], x[j], sigma)
	return omega

# ==============================================LTI=================================================#
# ==============================================END=================================================#
