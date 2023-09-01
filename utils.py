from ultralytics import YOLO

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

if __name__ == "__main__":
    results=predictions(model, source)
    