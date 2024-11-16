import optuna
import os
import wandb
from ultralytics import YOLO  

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="yolo11n_finetune"
    ,name="yolo11n_finetuned"
    # Track hyperparameters and run metadata
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# define the model
model = YOLO("comare_resutls/yolo11n_0.6435/weights/best.pt").to('cuda')

home = os.getcwd()
dataset_name = "maize-uav-crop-disease"

# Evaluate the model and return a metric (e.g., mAP)
metrics = model.val(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,project="yolo11n_finetune"
    ,name="yolo11n_finetuned"
    ,device = "0,1"
)
model.save("yolo11n-seg_trained_best.pt")
