import wandb
import os
from ray import tune
from ultralytics import YOLO  # Assuming this is the model you're using


wandb.login()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Load a YOLO11n model
model = YOLO("yolo11n-seg.pt") #.to('cuda')

home = os.getcwd()
dataset_name = "maize-uav-crop-disease"
    
# declare custom search space
# this is based on w


# Start tuning hyperparameters for YOLO11n training on the custom dataset
result_grid = model.tune(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    , use_ray=True
    , epochs=1
    , device = "0,1"
)
