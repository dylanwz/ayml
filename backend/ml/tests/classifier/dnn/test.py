from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot

from models.INetwork import *
from backend.services.classifier.dnn import *
from utils.math import *
from utils.functions import *

XarchitectureParams = {
    "networkShape": [784, 128, 10],
    "activation": "relu",
    "outputActivation": "sigmoid",
    "regularisation": "none",
    "initZero": False 
}

XtrainingParams = {
    "batchSize": 1,
    "learningRate": 0.1,
    "regLambda": 0,
    "lossFn": "square"
}

class DictAsObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

architectureParams = DictAsObject(XarchitectureParams)
trainingParams = DictAsObject(XtrainingParams)

classifier = ClassicClassifier(architectureParams, trainingParams)
# printNetwork(classifier.network)

# arr1 = [1,2,3]
# arr2 = [4,5,6]
# loss = 0
# lossFn = functionFactory('square')
# print(lossFn)
# for i in range(0,3):
#     print(lossFn.der(arr1[i], arr2[i], 3))

(Ntrain_X, train_y), (Ntest_X, test_y) = mnist.load_data()

train_X = np.array([np.ravel(x) for x in Ntrain_X])
test_X = np.array([np.ravel(x) for x in Ntest_X])

for i in range(0, 3000):
    print(f"Training: {i}")
    inp = [x/255 for x in train_X[i]]
    # printNetwork(classifier.network)
    classifier.tick(inp, ltoa(train_y[i], "mnist"))
    # printNetwork(classifier.network)

correct = 0
total = 0

resf = []
for j in range(0, 100):
    inp = [x/255 for x in test_X[j]]
    res2 = classifier.predict(inp)
    res = [ressy.outputVal for ressy in res2]
    print(res)
    ans = np.argmax(res)
    print(ans)
    print(test_y[j])
    if (ans == test_y[j]):
        correct += 1
    total += 1
    resf = res
    print(f"Test: {j}")

# printWeights(classifier.network)
print(resf)
print(correct/total)

# Can provide 92% acc