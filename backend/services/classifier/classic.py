from models.INetwork import *
from ml.classifier.handwritten.nn import *
    
class ClassicClassifier:
    def __init__(self, architecture: Architecture, trainingParams):
        self.batchSize = trainingParams.batchSize
        self.learningRate = trainingParams.learningRate
        self.regLambda = trainingParams.regLambda
        self.lossFn = trainingParams.lossFn
        network = buildNetwork(architecture.networkShape,
                               architecture.activation,
                               architecture.outputActivation,
                               architecture.regularisation,
                               architecture.initZero)
        self.network = network
        self.iters = 0
        self.epochs = 0
        self.loss = -1

    def tick(self, label):
        self.loss += backProp(self.network, label, self.lossFn)
        self.iters += 1;
        if (self.iters % self.batchSize == 0): 
            updateParams(self.network, self.learningRate, self.regLambda)
            self.epochs += 1
        return {"epochs": self.epochs, "loss": self.loss}
    
    def predict(self, inputs):
        return forwardProp(self.network, inputs)