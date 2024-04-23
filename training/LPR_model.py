'''
Training to detact license plates from LRP.yolov8 images 
https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4
'''
# import YOLO libary
from ultralytics import YOLO

# load the model
model = YOLO("yolov8n.yaml") # build a new model from scratch

if __name__ == '__main__':
    # train the model, if GPU is avaible then it will be used
    results = model.train(data="training\LPR.yolov8\config.yaml", epochs=300)

    # this is for M1, M2
    #results = model.train(data="training\LPR.yolov8\config.yaml", epochs=300, device='mps')