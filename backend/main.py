from typing import Union
from fastapi import FastAPI

from models.INetwork import *
# from ml.classifier.nn import *
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/network/run")
def run_network(network: Network):
    return {"item_id": item_id, "q": q}