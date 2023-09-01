from ultralytics import YOLO
import numpy as np

model = YOLO("runs/detect/train4/weights/best.pt")
source="IMG_7127.mov"

def predictions(model, source, show=True):
    results = model.predict(source=source, show=show)
    return results

def get_idx_class(class_name, classes):
    '''
    Args
    class_name: input name of the class
    classes: names of classes from yolo prediction 

    Return
    index of input class_name
    '''
    idx = list(classes.keys())
    classes = list(classes.values())
    position = classes.index(class_name)
    index_class = idx[position]
    return index_class

def angle_cal(x, y):
    '''
    Args:
     x: distance from x1 to x2
     y: distance from y1 to y2
    Return:
     degree which starts from 6 o'clock
    '''
    # Calculate the angle in radians using arctan2
    angle_rad = np.arctan2(y, x)

    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    # Adjust the angle to have 0 degrees at 6 o'clock (90 degrees)
    angle_deg = (angle_deg + 90) % 360

    return round(angle_deg)

if __name__ == "__main__":
    results=predictions(model, source)
    