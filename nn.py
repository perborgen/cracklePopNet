import numpy as np
import random
from random import randint

np.random.seed(4)

INSTANCES = 1000
BATCHES = 100
BATCH_SIZE = INSTANCES / BATCHES
ALPHA = 0.5
TEST_SIZE = 100

cracklePopObject = {
	0: 'CracklePop',
	1: 'Pop',
	2: 'Crackle'
}

def crossEntropyGradient(y, y_hat):
	return y_hat - y

def sigmoid(x):
	output = 1. / (1 + np.exp(-x))
	return output

def sigmoidDerivative(x):
	return sigmoid(x)*(1 - sigmoid(x))

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
		label = numToLabel(number)
		y.append(label)
	return X, y

X, y = createDataSet(INSTANCES)
X_test, y_test = createDataSet(TEST_SIZE)
w_0 = 2 * np.random.random((100, 10)) - 1
w_1 = 2 * np.random.random((10, 4)) - 1

def getBatch(X, y, number, size):
	start = number * size
	stop = (number + 1) * size
	return X[start:stop], y[start:stop]

def feed_forward(X, w_0, w_1):
	hidden_layer = sigmoid(np.dot(X, w_0))
	pred_input = np.dot(hidden_layer, w_1)
	prediction = softmax(pred_input)
	return prediction, hidden_layer

def backpropagation(X, y, prediction, hidden_layer):
	delta = crossEntropyGradient(y, prediction)
	w_1_grad = np.dot(hidden_layer.T, delta)
	hidden_delta = np.dot(delta, w_1.T) * sigmoidDerivative(hidden_layer)
	w_0_grad = np.dot(X.T, hidden_delta)
	return w_0_grad, w_1_grad

def test(X, y, w_0, w_1):
	prediction, _ = feed_forward(X, w_0, w_1)
	pred = np.argmax(prediction, 1)
	labels = np.argmax(y, 1)
	score = np.equal(pred, labels)
	score = score.astype(int)
	return np.mean(score)

for i in range(BATCHES):
	X_batch, y_batch = getBatch(X, y, i, BATCH_SIZE)
	prediction, hidden_layer = feed_forward(X_batch, w_0, w_1)
	w_0_grad, w_1_grad = backpropagation(X_batch, y_batch, prediction, hidden_layer)
	w_0 -= ALPHA * w_0_grad
	w_1 -= ALPHA * w_1_grad

score = test(X_test, y_test, w_0, w_1)
print 'Score: ', score

def numToFeatureVector(number):
	vector = np.zeros(100)
	vector[number - 1] = 1
	return np.array([vector]) 

def getValue(number, prediction):
	value = np.argmax(prediction, 1)[0]
	if value == 3:
		return number
	return cracklePopObject[value]

def runCracklePopNet(number):
	inputVector = numToFeatureVector(number)
	pred, _ = feed_forward(inputVector, w_0, w_1)
	print getValue(number, pred)


runCracklePopNet(14)
