from ultralytics import YOLO
from utils import predictions, get_idx_class
import cv2
import math
import time
import os

model = YOLO("runs/detect/train4/weights/best.pt")
source="IMG_7129.mov"
cap = cv2.VideoCapture(source)


while(cap.isOpened()):

    haveFrame, im = cap.read()

    if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    # get prediction
    preds = predictions(model, im, show=False)

    # information from predictions
    for result in preds:
        boxes = result.boxes
        names = result.names
    print(names)

    # if "tip" in names.values() and "base" in names.values():
    if len(boxes) > 1:
        # get coordinate x,y
        xywh = boxes.xywh.cpu().detach().numpy()

        tip = xywh[get_idx_class("tip", names)]
        base = xywh[get_idx_class("base", names)]

        x_t, y_t, w, h = tip
        x_b, y_b, w_b, h_b = base

        cv2.circle(im, (int(x_b), int(y_b)), 5, (0,255,0), -1)
        cv2.circle(im, (int(x_t), int(y_t)), 10, (0,255,0), 2)

    cv2.imshow('detect', im)
cap.release()
cv2.destroyAllWindows()

    

