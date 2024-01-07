from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.INetwork import *
from services.classifier.classic import *

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
def start_network(serviceParams: IServiceParams):
    serviceID = "S"+str(len(registry))
    registry[serviceID] = ClassicClassifier(serviceParams.architectureParams, serviceParams.trainingParams)
    return {"serviceID": serviceID}

@app.post("/classifier/classic/run")
def run_network(runParams: IRunParams):
    service = registry[runParams.serviceID]
    service.tick(runParams.inputs, runParams.labels)
    return {"epochs": service.epochs, "loss": service.loss}

# Jank solution for now
registry = {}
