import numpy as np
import random
from random import randint

np.random.seed(3)

INSTANCES = 2000
BATCHES = 100
BATCH_SIZE = INSTANCES / BATCHES
ALPHA = 0.5
TEST_SIZE = 200

cracklePopObject = {
	0: 'CracklePop',
	1: 'Pop',
	2: 'Crackle'
}

def crossEntropyGradient(y, y_hat):
	return y_hat - y

def softmax(x):
	x = np.exp(x)
	row_sum = np.sum(x, axis = 1)
	x /= row_sum.reshape((x.shape[0], 1))
	return x

def numToLabel(number):
	if number % 3 == 0 and number % 5 == 0:
		return [1, 0, 0, 0]
	elif number % 5 == 0:
		return [0, 1, 0, 0]
	elif number % 3 == 0:
		return [0, 0, 1, 0]
	else:
		return [0, 0, 0, 1]

def createDataSet(instances):
	y = []
	input_numbers = np.random.randint(1, 101, size=[instances])
	X = np.zeros((instances, 100))
	X[np.arange(instances), input_numbers - 1] = 1
	for number in input_numbers:
		y.append(numToLabel(number))
	return X, y

def getBatch(X, y, number, size):
	start = number * size
	stop = (number + 1) * size
	return X[start:stop], y[start:stop]

def feedForward(X, W):
	prediction = softmax(np.dot(X, W))
	return prediction

def backpropagation(X, y, prediction):
	delta = crossEntropyGradient(y, prediction)
	W_grad = np.dot(X.T, delta)
	return W_grad

def numToFeatureVector(number):
	vector = np.zeros(100)
	vector[number - 1] = 1
	return np.array([vector]) 

def getValue(number, prediction):
	value = np.argmax(prediction, 1)[0]
	if value == 3:
		return number
	return cracklePopObject[value]

def cracklePop(number):
	inputVector = numToFeatureVector(number)
	pred = feedForward(inputVector, W)
	print getValue(number, pred)


# Initialize training data, testing data and weights
X, y = createDataSet(INSTANCES)
X_test, y_test = createDataSet(TEST_SIZE)
W = 2 * np.random.random((100, 4)) - 1

# train the network in batches
for i in range(BATCHES):
	X_batch, y_batch = getBatch(X, y, i, BATCH_SIZE)
	prediction = feedForward(X_batch, W)
	W_grad = backpropagation(X_batch, y_batch, prediction)
	W -= ALPHA * W_grad

for i in range(1, 101):
	cracklePop(i)

