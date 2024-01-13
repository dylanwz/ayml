from models.INetwork import *
from backend.ml.classifier.dnn.dnn import *
from utils.functions import *

class ClassicClassifier:
    def __init__(self, architecture: IArchitecture, trainingParams: ITraining):
        self.batchSize = trainingParams.batchSize
        self.learningRate = trainingParams.learningRate
        self.regLambda = trainingParams.regLambda
        self.lossFn = functionFactory(trainingParams.lossFn)
        network = buildNetwork(architecture.networkShape,
                               functionFactory(architecture.activation),
                               functionFactory(architecture.outputActivation),
                               functionFactory(architecture.regularisation),
                               architecture.initZero)
        self.network = network
        self.iters = 0
        self.epochs = 0
        self.loss = -1

    def tick(self, inputs, label):
        for i in range(0, self.batchSize):
            self.unitTick(inputs, label)

    def unitTick(self, inputs, label):
        forwardProp(self.network, inputs)
        self.loss = backProp(self.network, label, self.lossFn)
        self.iters += 1
        if (self.iters % self.batchSize == 0): 
            updateParams(self.network, self.learningRate, self.regLambda)
            self.epochs += 1
        return {"epochs": self.epochs, "loss": self.loss}
    
    def predict(self, inputs):
        return forwardProp(self.network, inputs)