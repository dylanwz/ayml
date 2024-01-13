from typing import Callable, List
from pydantic import BaseModel

class IWire(BaseModel):
    weight: float
    errorDelta: float
    accErrorDelta: float
    regularisation: Callable

class INeuron(BaseModel):
    bias: float
    inputs: List[IWire]
    outputs: List[IWire]
    inputVal: float
    outputVal: float
    intputDelta: float
    outputDelta: float
    accDelta: float
    numAccumulatedDelta: float
    update: Callable

class IArchitecture(BaseModel):
    networkShape: List[int]
    activation: str
    outputActivation: str
    regularisation: str
    initZero: bool

class ITraining(BaseModel):
    batchSize: int
    learningRate: float
    regLambda: float
    lossFn: str

class IBuildParams(BaseModel):
    architectureParams: IArchitecture
    trainingParams: ITraining

class IRunParams(BaseModel):
    serviceID: str
    dataspace: str
