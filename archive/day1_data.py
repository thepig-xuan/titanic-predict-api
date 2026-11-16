import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
df = sns.load_dataset("titanic")
df["age"] = df["age"].fillna(df["age"].median())
df["sex"] = df["sex"].map({"female": 0, "male": 1})
print(df.head())

X = df[["pclass", "sex", "age", "fare"]].values
y = df["survived"].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

X_train_t = torch.tensor(X_train,dtype = torch.float32)
X_test_t = torch.tensor(X_test,dtype = torch.float32)
y_train_t = torch.tensor(y_train,dtype= torch.float32)
y_test_t = torch.tensor(y_test,dtype= torch.float32)


class TitanicDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y= y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]

train_dataset = TitanicDataset(X_train_t,y_train_t)
test_dataset = TitanicDataset(X_test_t,y_test_t)

train_loader = DataLoader(dataset = train_dataset,batch_size = 32,shuffle = True)
test_loader = DataLoader(dataset = test_dataset,batch_size = 32,shuffle = False)

for batch_X, batch_y in train_loader:
    print(batch_X.shape)
    print(batch_y.shape)
    break