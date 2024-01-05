from models.INetwork import *
from ml.classifier import *

class ClassifierService:
    def __init__(self, network: Network):
        self.architecture = network.architecture
        self.training = network.train 
    
    def model(self, architecture: Architecture):
        self.model = Network(architecture.shape, 
                             architecture.activation, 
                             architecture.outputActivation)

    def train(self, train: Train):
        self.model.train(train.learningRate, 
                         train.lossFunction, 
                         train.epochs, 
                         train.batchSize, 
                         train.validationSplit)

    def predict(self, inputs):
        return self.model.predict(inputs)