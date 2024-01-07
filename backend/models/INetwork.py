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

class IServiceParams(BaseModel):
    architectureParams: ArchitectureParams
    trainingParams: TrainingParams

class IRunParams(BaseModel):
    serviceID: str
    inputs: List[float]
    labels: List[float]
