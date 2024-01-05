from typing import Union
from fastapi import FastAPI

from models.INetwork import *
from services.classifier.classic import *

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/classifier/classic/start")
def start_network(architecture, trainingParams):
    return {"service": ClassicClassifier(architecture, trainingParams)}

@app.get("/classifier/classic/run")
def run_network(service, label):
    return service.tick(label)
