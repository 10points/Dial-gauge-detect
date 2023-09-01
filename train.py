from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Training
model.train(data="datasets/data.yaml", epochs=160) 