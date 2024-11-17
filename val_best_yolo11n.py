import optuna
import os
import wandb
from ultralytics import YOLO  

# wandb.login()
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="yolo11nGYMNSA"
#     ,name="yolo11nGYMNSA"
#     # Track hyperparameters and run metadata
# )


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# define the model
model = YOLO("yolo11n_trained_GYMNSA.pt").to('cuda')

home = os.getcwd()
dataset_name = "GYMNSA"

# Evaluate the model and return a metric (e.g., mAP)
metrics = model.val(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    # ,project="yolo11nGYMNSA"
    # ,name="yolo11n_finetuned"
    ,device = "0"
)
