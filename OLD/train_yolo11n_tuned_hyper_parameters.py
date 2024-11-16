import optuna
import os
import wandb
from ultralytics import YOLO  

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="yolo11n_finetune",
    # Track hyperparameters and run metadata
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# define the model
model = YOLO("yolo11x-seg.pt").to('cuda')

lr0 = 0.01
epochs = 100  
# iou = 0.6    
weight_decay = 0.00042
momentum = 0.9372
batch = 64
optimizer = "auto"    

home = os.getcwd()
dataset_name = "maize-uav-crop-disease"

# Initialize YOLOv11 model
model.train(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    # ,lr0=lr0
    # ,iou=iou
    # ,batch=batch
    # ,momentum=momentum
    # ,weight_decay=weight_decay
    # ,epochs = epochs
    # ,optimizer=optimizer
    # ,iterations=
    ,project="yolo11n_finetune"
    ,name="yolo11x"
    ,device = "0,1"
    # ,patience = 5
    # ,save = True
    # ,save_period = 5
    # ,cache = True
    # ,close_mosaic = 0
)

# Evaluate the model and return a metric (e.g., mAP)
metrics = model.val(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,device = "0,1"
)
model.save("yolo11x-seg_trained.pt")
