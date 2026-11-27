import torch
from config import FEATURES
from model import load_model

model = load_model()

p = {"pclass": 1, "sex": 0, "age": 30, "fare": 100, "parch": 1, "sibsp": 0}
x = torch.tensor([[p[name] for name in FEATURES]], dtype=torch.float32)

with torch.no_grad():
    logit = model(x)
    prob = torch.sigmoid(logit)

print(f"原始分数：{logit.item():.4f}")
print(f"存货概率：{prob.item():.4f}")


