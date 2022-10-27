import numpy as np
import pandas as pd
from sklearn import preprocessing


# 1.3.1
def n1():
	Z = np.zeros((8, 8), dtype=int)
	Z[1::2, ::2] = 1
	Z[::2, 1::2] = 1
	print(Z)


# n1()


# 1.3.2
def n2():
	Z = np.zeros((5, 5))
	Z += np.arange(5)
	print(Z)


# n2()


# 1.3.3
def n3():
	Z = np.random.random((3, 3, 3))
	print(Z)


# n3()


# 1.3.4
def n4():
	Z = np.ones((5, 5))
	Z[1:-1, 1:-1] = 0
	print(Z)


# n4()


# 1.3.5
def n5():
	Z = np.random.random((10))
	Z.sort()
	print(Z)


# n5()


# 1.3.6
def n6():
	Z = np.ones((3, 5))
	print("Размерность -", Z.size)
	print("Размеры -", Z.shape)


# n6()


# 2.3.1
def n7():
	a = pd.Series([1, 2, 3, 4])
	b = pd.Series([1, 0, 2, 6])

	res = np.linalg.norm(a - b)
	print(res)


# n7()


# 2.3.2
# 2.3.3
def n8():
	url = "https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv"
	dataFrame = pd.read_csv(url)
	print(dataFrame.head(5))
	print(dataFrame.tail(2))
	print(dataFrame.shape)
	print(dataFrame.describe())
	print(dataFrame.iloc[1:2])
	print(dataFrame[dataFrame['Age'] < 2].head(5))


# n8()


# 3.3.2
def n9():
	url = "https://raw.githubusercontent.com/akmand/datasets/master/iris.csv"
	dataFrame = pd.read_csv(url)
	scaler = preprocessing.MinMaxScaler()
	dataFrame[['sepal_length_cm']] = scaler.fit_transform(dataFrame[['sepal_length_cm']])
	scaler = preprocessing.StandardScaler()
	dataFrame[['sepal_width_cm']] = scaler.fit_transform(dataFrame[['sepal_width_cm']])
	print(dataFrame)


n9()
