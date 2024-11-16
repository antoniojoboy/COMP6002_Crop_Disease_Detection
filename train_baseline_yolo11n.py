"""
This is to estabilish a baseline for the maize 
"""

from ultralytics import YOLO
import os
import wandb

home = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(home)

model_name = "yolov8n-seg" # segmentation
dataset_name = "leaf-disease"

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="yolov8n_"+dataset_name
    ,name="yolov8n"
    # Track hyperparameters and run metadata
)


# Load a yolov8n model
model = YOLO("yolov8n-seg.pt").to("cuda")

# Start tuning hyperparameters for yolov8n training on the COCO8 dataset
result_grid = model.train(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,project="yolov8n_"+dataset_name
    ,name="yolov8n"
    ,device = "0,1"
    ,epochs = 100
    )

model.save(model_name+"_trained_"+dataset_name+".pt")
