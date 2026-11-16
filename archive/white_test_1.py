import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

df = sns.load_dataset("titanic")
df['age'] =df['age'].fillna(df['age'].median())
df['sex'] = df['sex'].map({'male':1,'female':0})

X = df[["sex","age","pclass","fare"]].values
y = df["survived"].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

X_train_t = torch.tensor(X_train,dtype = torch.float32)
X_test_t = torch.tensor(X_test,dtype = torch.float32)
y_train_t = torch.tensor(y_train,dtype = torch.float32)
y_test_t = torch.tensor(y_test,dtype = torch.float32)

class TitanicDataset(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

train_dataset = TitanicDataset(X_train_t,y_train_t)
test_dataset = TitanicDataset(X_test_t,y_test_t)


train_loader = DataLoader(train_dataset,batch_size = 32,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = 32,shuffle = False)

class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4,16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16,1)
    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
model = TitanicModel()

epochs = 20
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x,batch_y in train_loader:
        y_pred = model(batch_x)
        y_pred  = y_pred.squeeze()
        loss = loss_fn(y_pred,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"epoch:{epoch+1},loss:{avg_loss:.4f}")

model.eval()
total = 0
correct = 0
with torch.no_grad():
    for batch_x,batch_y in test_loader:
        y_pred = model(batch_x)
        y_pred = y_pred.squeeze()

        predicted = (y_pred > 0).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
accuracy = correct / total
print(f"模型准确率:{accuracy:.4f}")

