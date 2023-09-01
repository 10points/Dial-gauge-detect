from roboflow import Roboflow
import os

SECRET_KEY = os.environ.get("SECRET_KEY")
rf = Roboflow(api_key=SECRET_KEY)
project = rf.workspace("myproject-dua5p").project("gauge-detection-5l4ey")
dataset = project.version(2).download("yolov8", location="datasets")




