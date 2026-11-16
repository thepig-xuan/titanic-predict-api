import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel,Field
from config import FEATURES

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = TitanicModel()
model.load_state_dict(torch.load("titanic_model.pt"))
model.eval()

app = FastAPI()


class Passenger(BaseModel):
    pclass: int = Field(ge=1, le=3, description="舱位：1/2/3")
    sex: int = Field(ge=0, le=1, description="0=女，1=男")
    age: float = Field(ge=0, le=120, description="年龄")
    fare: float = Field(ge=0, description="票价")
    sibsp: int = Field(default=0, ge=0, le=10)
    parch: int = Field(default=0, ge=0, le=10)


@app.post("/predict")
def predict(p: Passenger):
    values = [getattr(p, name) for name in FEATURES]
    x = torch.tensor([values], dtype=torch.float32)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    return {
        "survived_prob" : round(prob,4),
         "verdict": "存活概率高" if prob > 0.5 else "存活概率低"
    }
