from pydantic import BaseModel

class Architecture(BaseModel):
    shape: list
    activation: str
    outputActivation: str

class Train(BaseModel):
    learningRate: float
    lossFunction: str
    epochs: int
    batchSize: int 
    validationSplit: float

class Network(BaseModel):
    architecture: Architecture
    train: Train