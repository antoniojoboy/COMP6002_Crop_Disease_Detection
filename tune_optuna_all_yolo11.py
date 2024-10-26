import optuna
import os
import wandb
from ultralytics import YOLO  

wandb.login()

name = ""
project="all_YOLO11",

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def objective(trial):
    # Define hyperparameters based on the trial
    lr0 = trial.suggest_loguniform('learning_rate', 0.001, 0.99)
    batch = int(trial.suggest_discrete_uniform('batch', 4, 100,4))
    weight_decay = trial.suggest_loguniform('weight_decay',0.0001,0.9)
    momentum = trial.suggest_uniform('momentum', 0.001, 0.99)
    # epochs = int(trial.suggest_discrete_uniform('epochs', 50, 100,10))
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    # define the model
    model = YOLO(name).to('cuda')
    # model.load("YOLO11n-seg_trained_maize-disease-20240221-8.pt")
    
    home = os.getcwd()
    epochs = 2
    dataset_name = "maize-uav-crop-disease"
    
    # Initialize YOLOv11 model
    model.tune(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,lr0=lr0
        ,batch=batch
        ,momentum=momentum
        ,weight_decay=weight_decay
        ,epochs=epochs
        ,optimizer=optimizer
        ,project=project
        ,name=name
        ,device = "0,1"
        # ,patience = 5
        ,save = True
        ,save_period = 10
        ,cache = True
    )
    
    # Evaluate the model and return a metric (e.g., mAP)
    metrics = model.val(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,device = "0,1"
    )
    
    mAP = metrics.seg.map
    return mAP  # Higher is better in this case


def run_all():
    model_names = [
        "yolo11n-seg.pt"
        ,"yolo11s-seg.pt"
        ,"yolo11m-seg.pt"
        ,"yolo11l-seg.pt"
        ,"yolo11x-seg.pt"
    ]
    
    global name
    global project
    project="all_YOLO11"
    
    for model_name in model_names:
        name = model_name
        study = optuna.create_study(direction='maximize')  # We want to maximize the mAP
        study.optimize(objective, n_trials=2)  # Run for 50 trials (or more based on resources)

        # Get the best hyperparameters
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)

run_all()