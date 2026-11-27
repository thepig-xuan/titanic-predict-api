import torch

ckpt = torch.load("titanic_model.pt")
print(ckpt.keys())
print(ckpt["features"])