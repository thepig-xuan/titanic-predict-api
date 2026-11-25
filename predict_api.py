from typing import List
from pathlib import Path
from fastapi import FastAPI
import torch
import torch.nn as nn
from pydantic import BaseModel,Field
from config import FEATURES

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6,16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16,1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

MODEL_PATH = Path(__file__).parent / "titanic_model.pt"
model = TitanicModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

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
        "prob" :prob.item(),
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
