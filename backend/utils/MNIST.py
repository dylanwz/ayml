import numpy as np
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = np.array([np.ravel(x) for x in train_X])
test_X = np.array([np.ravel(x) for x in test_X])

print("MNIST processing")

def getMNISTTrain(i):
    return {"input": train_X[i], "label": train_y[i]}

def getMNISTTest(i):
    return {"input": test_X[i], "label": test_y[i]}