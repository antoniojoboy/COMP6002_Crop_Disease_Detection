"""
This is to estabilish a baseline for the maize 
"""

from ultralytics import YOLO
import os

home = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(home)

model_name = "yolo11n" # segmentation
dataset_name = "maize-uav-crop-disease"

# Load a YOLO11n model
model = YOLO("yolo11n-seg.pt")

# Start tuning hyperparameters for YOLO11n training on the COCO8 dataset
result_grid = model.train(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,device = "0,1"
    ,epochs = 100
    )

model.save(model_name+"_trained_"+dataset_name+".pt")
