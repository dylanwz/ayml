from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.INetwork import *
from services.visual.dnn import *
from utils.dataspaces import *
from utils.math import *

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/classifier/classic/start")
def start_network(buildParams: IBuildParams):
    serviceID = "S"+str(len(registry))
    registry[serviceID] = VisualDNNService(buildParams.architectureParams, buildParams.trainingParams)
    return {"serviceID": serviceID}

@app.post("/classifier/classic/run")
def run_network(runParams: IRunParams):
    service = registry[runParams.serviceID]
    input = getTrainInput(runParams.dataspace, service.iters)
    label = ltoa(getTrainLabel(runParams.dataspace, service.iters), runParams.dataspace)
    service.tick(input, label)
    return {"epochs": service.epochs, "loss": service.loss}

# Jank solution for now
registry = {}
