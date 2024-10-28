import optuna
import os
import wandb
from ultralytics import YOLO  

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="yolo11n_best",
    # Track hyperparameters and run metadata
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# def objective(trial):
#     # Define hyperparameters based on the trial
#     # lr0 = trial.suggest_loguniform('learning_rate', 0.009, 0.011)
#     # batch = int(trial.suggest_discrete_uniform('batch', 4, 100,4))
#     # weight_decay = trial.suggest_loguniform('weight_decay',0.0004,0.0009)
#     # momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
#     # optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

#     # define the model
#     model = YOLO("yolo11n-seg.pt").to('cuda')
#     model = YOLO("yolo11n-seg.pt").to('cuda')
    
#     lr0 = 0.01    
#     epochs = 100  
#     iou = 0.7
#     weight_decay = 0.0005
#     momentum = 0.937
#     batch = 32
#     iterations = 100
#     optimizer = "SGD"    
#     home = os.getcwd()
#     dataset_name = "maize-uav-crop-disease"
    
#     # Initialize YOLOv11 model
#     model.tune(
#         data=home+"/dataset/"+dataset_name+"/data.yaml"
#         ,lr0=lr0
#         ,iou=iou
#         ,batch=batch
#         ,momentum=momentum
#         ,weight_decay=weight_decay
#         ,epochs = epochs
#         ,optimizer=optimizer
#         ,iterations=iterations
#         ,project="yolo11n_best"
#         ,name="yolo11n"
#         ,device = "0,1"
#         # ,patience = 5
#         ,save = True
#         # ,save_period = 5
#         ,cache = True

#     )

#     # Evaluate the model and return a metric (e.g., mAP)
#     metrics = model.val(
#         data=home+"/dataset/"+dataset_name+"/data.yaml"
#         ,device = "0,1"
#     )
    
#     mAP = metrics.seg.map
#     return mAP  # Higher is better in this case

# study = optuna.create_study(direction='maximize')  # We want to maximize the mAP
# study.optimize(objective, n_trials=50)  # Run for 50 trials (or more based on resources)

# # Get the best hyperparameters
# best_params = study.best_params
# print("Best Hyperparameters:", best_params)

model = YOLO("yolo11n-seg.pt").to('cuda')

lr0 = 0.01    
epochs = 100  
iou = 0.7
weight_decay = 0.0005
momentum = 0.937
batch = 64
iterations = 100000
optimizer = "SGD" 
home = os.getcwd()
dataset_name = "maize-uav-crop-disease"

# Initialize YOLOv11 model
model.tune(
    data=home+"/dataset/"+dataset_name+"/data.yaml"
    # ,lr0=lr0
    # ,iou=iou
    ,batch=batch
    # ,momentum=momentum
    # ,weight_decay=weight_decay
    # ,optimizer=optimizer
    # ,epochs = epochs
    ,iterations=iterations
    ,project="yolo11n_best"
    ,name="yolo11n"
    ,device = "0,1"
    # ,patience = 5
    # ,save = True
    # ,save_period = 5
    # ,cache = True

)