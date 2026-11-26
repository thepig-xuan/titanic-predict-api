import torch
from predict_api import model,FEATURES,app
from fastapi.testclient import TestClient


def test_model_output_shape():
    x = torch.randn(3,len(FEATURES))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3,1)

def test_features_match_model_input():
    assert len(FEATURES) == model.layer1.in_features

client = TestClient(app)

def test_known_passenger():
    resp = client.post("/predict_one", json={"pclass": 2, "sex": 0, "age": 25, "fare": 30, "parch": 1, "sibsp": 0},
    )
    assert resp.status_code == 200
    prob = resp.json()["prob"]
    assert abs(prob - 0.633263) < 0.0001

def test_batch_matches_single():
    p = {"pclass": 2, "sex": 0, "age": 25, "fare": 30, "parch": 1, "sibsp": 0}
    r1 = client.post("/predict_one", json=p).json()["prob"]
    # print("单条接口返回",r1.json())
    r2 = client.post("/predict_batch", json={"passengers":[p]}).json()["results"][0]
    # print("批量接口返回",r2.json())
    assert abs(r1-r2) < 0.0001