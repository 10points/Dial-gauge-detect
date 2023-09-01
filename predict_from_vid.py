from ultralytics import YOLO
from utils import predictions, get_idx_class, angle_cal
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
    # print(names)

   
    if len(boxes) > 1: # show prediction when both detection of base and tip occurs
        # get coordinate x,y
        xywh = boxes.xywh.cpu().detach().numpy()

        tip = xywh[get_idx_class("tip", names)]
        base = xywh[get_idx_class("base", names)]

        x_t, y_t, w, h = tip
        x_b, y_b, w_b, h_b = base

        # Angle cal
        # coordinate of base - coordinate of tip
        dx = x_b - x_t
        dy = y_b - y_t

        # get an angle
        theta = angle_cal(dx, dy)

        # psi cal
        value = 100.46*theta - 4194.9

        cv2.circle(im, (int(x_b), int(y_b)), 5, (0,255,0), -1)
        cv2.circle(im, (int(x_t), int(y_t)), 10, (0,255,0), 1)
        cv2.putText(im, f"{value:.2f} psi", (int(x_t)-20, int(y_t)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow('detect', im)
cap.release()
cv2.destroyAllWindows()

    

