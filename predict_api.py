from typing import List

from fastapi import FastAPI
import torch
from model import load_model
from pydantic import BaseModel,Field
from config import FEATURES


model = load_model()


app = FastAPI()
class Titanic(BaseModel):
    pclass: int = Field(ge=1, le=3, description="舱位：1/2/3")
    sex: int = Field(ge=0, le=1, description="0=女，1=男")
    age: float = Field(ge=0, le=120, description="年龄")
    fare: float = Field(ge=0, description="票价")
    sibsp: int = Field(default=0, ge=0, le=10)
    parch: int = Field(default=0, ge=0, le=10)

class BatchRequest(BaseModel):
    passengers : List[Titanic] = Field(min_length=1,max_length=100)

@app.post("/predict_one")
def predict_one(t : Titanic):
    values = [getattr(t, name) for name in FEATURES]
    x = torch.tensor([values],dtype=torch.float32)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit)

    return{
        "prob" :round(prob.item(),4)
    }
@app.post("/predict_batch")
def predict_batch(req : BatchRequest):
    values = [[getattr(t, name) for name in FEATURES] for t in req.passengers]
    x = torch.tensor(values,dtype=torch.float32)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).flatten()

    return{
        "results": [round(p, 4) for p in prob.tolist()]
    }
