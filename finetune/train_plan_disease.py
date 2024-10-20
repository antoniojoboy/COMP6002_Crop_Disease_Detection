from ultralytics import YOLO
import os

home = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(home)

model_name = "YOLO11n"
model_name = "YOLO11n-seg" # segmentation
dataset_name = "-plantvillage-dataset-1"
dataset_name = "maize-disease-20240221-8"

# Load a YOLO11n model
model = YOLO("yolo11n-seg.pt")
# model.load(home+"/runs/detect/train6/weights/best.pt")

# Start tuning hyperparameters for YOLO11n training on the COCO8 dataset
# result_grid = model.tune(data="data.yaml")
result_grid = model.train(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,device = "0,1"
    ,epochs = 100
    )

model.save(model_name+"_trained_"+dataset_name+".pt")
# model.load(model_name+"_trained_"+dataset_name+".pt")