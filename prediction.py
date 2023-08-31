from ultralytics import YOLO

model = YOLO("Dial-gauge-detect/runs/detect/train3/weights/best.pt")
results = model.predict(source="Dial-gauge-detect/IMG_7127.mov", show=True)