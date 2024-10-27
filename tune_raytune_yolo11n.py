import optuna
import os
import wandb
from ultralytics import YOLO  

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="yolo11n_raytune",
    # Track hyperparameters and run metadata
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# define the model
model = YOLO("yolo11n-seg.pt")
home = os.getcwd()
dataset_name = "maize-uav-crop-disease"

# Initialize YOLOv11 model
result_grid  = model.tune(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,project="yolo11n_raytune"
    ,name="yolo11n"
    ,device = "0,1"
    ,use_ray=True
    ,save = True
    ,cache = True
)

# Evaluate the model and return a metric (e.g., mAP)
metrics = model.val(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,device = "0,1"
)

mAP = metrics.seg.map
print("Best mAP:", mAP)
