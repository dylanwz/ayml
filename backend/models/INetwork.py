from typing import Callable, List
from pydantic import BaseModel

class Function(BaseModel):
    out: Callable[[float], float]
    der: Callable[[float], float]

class Architecture(BaseModel):
    shape: list
    activation: str
    outputActivation: str
    regularisation: Function
    initZero: bool

class TrainingParams(BaseModel):
    batchSize: int
    learningRate: float
    regLambda: float
    lossFn: Function