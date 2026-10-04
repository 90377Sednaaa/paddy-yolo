import importlib
import os
import yaml


def test_imports():
    assert importlib.util.find_spec("torch") is not None
    assert importlib.util.find_spec("ultralytics") is not None
    assert importlib.util.find_spec("paddy_yolo") is not None
    assert importlib.util.find_spec("paddy_yolo.modules") is not None
    assert importlib.util.find_spec("paddy_yolo.loss") is not None
    assert importlib.util.find_spec("paddy_yolo.engine") is not None
    assert importlib.util.find_spec("paddy_yolo.dataset") is not None


def test_configs_exist():
    assert os.path.exists("configs/r2000.yaml")
    assert os.path.exists("configs/hyp_paper.yaml")
    assert os.path.exists("requirements.txt")


def test_r2000_config_content():
    with open("configs/r2000.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["nc"] == 16
    expected_classes = [
        "rice leaf roller",
        "pink rice borer",
        "asiatic rice borer",
        "yellow rice borer",
        "rice leaf caterpillar",
        "small brown plant hopper",
        "white backed plant hopper",
        "brown plant hopper",
        "rice leaf hopper",
        "rice gall midge",
        "paddy stem maggot",
        "rice stem fly",
        "grain spreader thrips",
        "rice weevil",
        "rice shell pest",
        "melanitis leda",
    ]
    if isinstance(data["names"], dict):
        names_list = [data["names"][i] for i in range(16)]
    else:
        names_list = data["names"]
    assert names_list == expected_classes


def test_hyp_paper_config_content():
    with open("configs/hyp_paper.yaml", "r", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)
    assert hyp["epochs"] == 300
    assert hyp["batch"] == 32
    assert hyp["imgsz"] == 640
    assert hyp["optimizer"] == "AdamW"
    assert hyp["lr0"] == 0.01
    assert hyp["lrf"] == 0.01
    assert hyp["momentum"] == 0.937
    assert hyp["weight_decay"] == 0.0005
    assert hyp["close_mosaic"] == 10
