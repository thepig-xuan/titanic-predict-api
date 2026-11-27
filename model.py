from pathlib import Path

import torch
import torch.nn as nn

from config import FEATURES

MODEL_PATH = Path(__file__).parent / "titanic_model.pt"

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(len(FEATURES), 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x  = self.layer2(x)
        return x

def load_model():
    """加载模型，并校验训练时的特征顺序与 config 是否一致"""
    ckpt = torch.load(MODEL_PATH)

    if ckpt["features"] != FEATURES:
        raise RuntimeError(
            f"模型是用 {ckpt['features']} 训练的，"
            f"但 config 现在是 {FEATURES}。请重新运行 train.py"
        )

    model = TitanicModel()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model