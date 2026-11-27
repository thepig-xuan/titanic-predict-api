import torch
import torch.nn as nn
from pathlib import Path

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6,16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16,1)
    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

MODEL_PATH = Path(__file__).parent / "titanic_model.pt"
model = TitanicModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


x = torch.tensor([[1.0, 0.0, 30.0, 100.0, 1.0, 0.0]])

with torch.no_grad():
    logit = model(x)
    prob = torch.sigmoid(logit)

print(f"原始分数：{logit.item()}")
print(f"存货概率：{prob.item()}")


