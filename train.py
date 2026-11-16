import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config import FEATURES

#生成工随机种子
torch.manual_seed(42)
np.random.seed(42)

#加载并清洗数据
df = sns.load_dataset("titanic")
df["age"] = df["age"].fillna(df["age"].median())
df["sex"] = df["sex"].map({"female": 0, "male": 1})


X = df[FEATURES].values
y = df["survived"].values

#划分测试及和训练集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#把数据转换成torch能接受的张量
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

train_loader = DataLoader(dataset = train_dataset,batch_size = 32,shuffle = True)#一次输出32组数据
test_loader = DataLoader(dataset = test_dataset,batch_size = 32,shuffle = False)



class TitanicModel(nn.Module):
    def __init__(self):#定义方法
        super().__init__()
        self.layer1 = nn.Linear(6,16)#算出来中间数据
        self.relu = nn.ReLU()#负数变成零
        self.layer2 = nn.Linear(16,1)#把layer1输出的16组w和b压成1组，就是最终的预测结果

    def forward(self, x):#定义执行顺序
        x = self.layer1(x)
        x = self.relu(x)#负数变成零
        x = self.layer2(x)
        return x

model = TitanicModel()
print(model)

loss_fn = nn.BCEWithLogitsLoss()#损失计算器（算出loss差距有多大）
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)#更新w和b#设置学习率



epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0  # 开考前，总分归零
    for batch_x,batch_y in train_loader:
        y_pred= model(batch_x)
        y_pred = y_pred.squeeze()#把y_pred数组【32，1】压成一维
        loss = loss_fn(y_pred,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 每做完一题，加进总分
    avg_loss = epoch_loss / len(train_loader)  # 考完，除以题数

    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")



#查看模型准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x,batch_y in test_loader:
        y_pred = model(batch_x)
        y_pred = y_pred.squeeze()

        predicted = (y_pred > 0).float()

        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = correct / total
print(f"模型最终的准确率：{accuracy*100:.2f}")


# 把刚才做好的模型拉出来，提取它做错的题目
model.eval()
wrong_cases = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        y_pred = model(batch_x).squeeze()
        predicted = (y_pred > 0).float()#也可以写为torch.sigmoid(y_pred) > 0.5

        wrong_idx = (predicted != batch_y).nonzero(as_tuple=True)[0]
        for idx in wrong_idx:
            x_val = batch_x[idx].tolist()
            real = int(batch_y[idx].item())
            pred = int(predicted[idx].item())
            wrong_cases.append((x_val,real,pred))

print(f"\n--- 模型预测错误的案发现场（共{len(wrong_cases)}个）---")
for i, (x, real, pred) in enumerate(wrong_cases[:5]):  # 只看前5个
    print(f"案发{i + 1}: 特征[舱位:{x[0]}, 性别:{x[1]}, 年龄:{x[2]:.1f}, 票价:{x[3]:.1f}] | 真实:{real}, 预测:{pred}")




torch.save(model.state_dict(),"titanic_model.pt")
print("模型已保存")