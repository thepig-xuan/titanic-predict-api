# 泰坦尼克生存预测 API

## 这是什么

输入"pclass", "sex", "age", "fare", "parch", "sibsp"
输出这个人的生存率

## 环境

- conda 环境：`ai_learning`
- 安装依赖：`pip install -r requirements.txt
-Python 3.x，建议用 conda 建独立环境

## 怎么跑

1. 训练并生成模型文件（跑完会多出 `titanic_model.pt`）

   `python train.py`

2. 启动服务（必须在项目根目录，也就是 `train.py` 所在的那一层）

   `uvicorn predict_api:app --reload`

   看到 `Application startup complete` 就是起来了

3. 浏览器打开 http://127.0.0.1:8000/docs


## 接口
{
  "passengers": [
    {"pclass": 1, "sex": 0, "age": 30, "fare": 100},
    {"pclass": 3, "sex": 1, "age": 22, "fare": 8}
  ]
}


复制粘贴进去测试

返回示例：
{
  "survived_prob": 0.8343,
  "verdict": "存活概率高"
}

## 自检（改特征顺序后必跑）

用例：`pclass=2, sex=0, age=25, fare=30, parch=1, sibsp=0`
（parch 和 sibsp 必须不同，否则顺序错了也看不出来）

1. `predict_demo.py` 跑一遍，记下概率
2. `/docs` 里传同样的参数，记下概率
3. 两个数必须完全一致
