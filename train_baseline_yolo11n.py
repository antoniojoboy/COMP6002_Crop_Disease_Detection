"""
This is to estabilish a baseline for the maize 
"""

from ultralytics import YOLO
import os
import wandb

home = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(home)

model_name = "yolo11n" # segmentation
dataset_name = "GYMNSA"

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="yolo11n_"+dataset_name
    ,name="yolo11n"
    # Track hyperparameters and run metadata
)


# Load a yolo11n model
model = YOLO(model_name+".pt").to("cuda")

# Start tuning hyperparameters for yolo11n training on the COCO8 dataset
result_grid = model.train(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    # data="coco128-seg.yaml"
    ,project="yolo11n_"+dataset_name
    ,name="yolo11n"
    ,workers = 0
    # ,device = "0"
    # ,epochs = 100
    )

model.save(model_name+"_trained_"+dataset_name+".pt")
