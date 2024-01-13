from utils.MNIST import *

def getTrainInput(dataspace: str, index: int):
    match dataspace.upper():
        case "MNIST":
            return getMNISTTrain(index)["input"]
        
def getTestInput(dataspace: str, index: int):
    match dataspace.upper():
        case "MNIST":
            return getMNISTTest(index)["input"]
        
def getTrainLabel(dataspace: str, index: int):
    match dataspace.upper():
        case "MNIST":
            return getMNISTTrain(index)["label"]
        
def getTestLabel(dataspace: str, index: int):
    match dataspace.upper():
        case "MNIST":
            return getMNISTTest(index)["label"]