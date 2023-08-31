from roboflow import Roboflow
from ultralytics import YOLO
import os

SECRET_KEY = os.environ.get("SECRET_KEY")
rf = Roboflow(api_key=SECRET_KEY)
project = rf.workspace("myproject-dua5p").project("gauge-detection-5l4ey")
dataset = project.version(1).download("yolov8", location="dataset")

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Training
model.train(data="data.yaml", epochs=200) 