from typing import Callable, List
from pydantic import BaseModel

class Wire(BaseModel):
    weight: float
    errorDelta: float
    accErrorDelta: float
    regularisation: Callable

class Neuron(BaseModel):
    bias: float
    inputs: List[Wire]
    outputs: List[Wire]
    inputVal: float
    outputVal: float
    intputDelta: float
    outputDelta: float
    accDelta: float
    numAccumulatedDelta: float
    update: Callable

class ArchitectureParams(BaseModel):
    networkShape: List[int]
    activation: str
    outputActivation: str
    regularisation: str
    initZero: bool

class TrainingParams(BaseModel):
    batchSize: int
    learningRate: float
    regLambda: float
    lossFn: str

class ServiceParams(BaseModel):
    architectureParams: ArchitectureParams
    trainingParams: TrainingParams

class IService(BaseModel):
    batchSize: int
    learningRate: float
    regLambda: float
    lossFn: str
    network: List[List[Neuron]]
    iters: int
    epochs: int
    loss: float

class Test(BaseModel):
    one: int