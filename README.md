# 泰坦尼克生存预测 API

## 这是什么

一个从训练到部署的完整示例：用 PyTorch 训练一个 MLP 预测泰坦尼克乘客的存活率，
保存模型，再用 FastAPI 提供预测接口。

输入：6 个乘客特征 → 输出：存活概率（0~1）
## 效果
- 测试集准确率 **80.45%**（891 条数据按 8:2 划分，测试集 179 条）
- 模型规模：**129 个参数**，文件 3 KB
- 单条预测耗时 < 5ms（CPU）

## 环境

- conda 环境：`ai_learning`
- 安装依赖：`pip install -r requirements.txt`
- Python 3.x，建议用 conda 建独立环境

## 怎么跑

1. 训练并生成模型文件（跑完会多出 `titanic_model.pt`）

   `python train.py`

2. 启动服务（必须在项目根目录，也就是 `train.py` 所在的那一层）

   `uvicorn predict_api:app --reload`

   看到 `Application startup complete` 就是起来了

3. 浏览器打开 http://127.0.0.1:8000/docs


## 接口

服务启动后有两个接口，都在 http://127.0.0.1:8000/docs 里可以直接试。


### POST /predict_one
请求：
{"pclass": 1, "sex": 0, "age": 30, "fare": 100}
返回：
{"prob": 0.8343}

### POST /predict_batch —— 一次预测多人
请求：
```json
{
  "passengers": [
    {"pclass": 1, "sex": 0, "age": 30, "fare": 100},
    {"pclass": 3, "sex": 1, "age": 22, "fare": 8}
  ]
}
```
返回：
```json
{"results": [0.8343, 0.1032]}
```

结果顺序与传入顺序一致


| 字段 | 含义 | 必填 |
|---|---|---|
| pclass | 舱位 1/2/3 | 是 |
| sex | 0=女，1=男 | 是 |
| age | 年龄 | 是 |
| fare | 票价 | 是 |
| sibsp | 同行兄弟姐妹/配偶数 | 否，默认 0 |
| parch | 同行父母/子女数 | 否，默认 0 |


## 自检（改特征顺序后必跑）

用例：`pclass=2, sex=0, age=25, fare=30, parch=1, sibsp=0`
（parch 和 sibsp 必须不同，否则顺序错了也看不出来）

1. `predict_demo.py` 跑一遍，记下概率
2. `/docs` 里传同样的参数，记下概率
3. 两个数必须完全一致
