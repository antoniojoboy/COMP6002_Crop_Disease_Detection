"""
This is to estabilish a baseline for the maize 
"""

from ultralytics import YOLO
import os

home = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(home)

model_name = "yolo11n-seg" # segmentation
dataset_name = "maize-uav-crop-disease"

# initialize hyper paramter values
lr0 = 0.09677
batch = 70
momentum = 0.70297
weight_decay = 0.00067
epochs = 1000
optimizer = "SGD"

# Load a YOLO11n model
model = YOLO("yolo11n-seg.pt")
 
model.tune(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,lr0=lr0
    ,batch=batch
    ,momentum=momentum
    ,weight_decay=weight_decay
    ,epochs = epochs
    ,optimizer=optimizer
    ,project="yolo11n_demo_test"
    ,name="yolo11n_optimized"
    ,patience = 30
    ,device = "0,1"
    ,save = True
    ,save_period = 5
    ,cache = True
)

metrics = model.val(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    ,device = "0,1"
)
    

model.save(model_name+"_trained_"+dataset_name+".pt")
# model.load(model_name+"_trained_"+dataset_name+".pt")