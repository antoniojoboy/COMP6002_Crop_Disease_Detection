import optuna
import os
import wandb
from ultralytics import YOLO  

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="optuna_maize_baysian",
    # Track hyperparameters and run metadata
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def objective(trial):
    # Define hyperparameters based on the trial
    lr0 = trial.suggest_loguniform('learning_rate', 0.09, 0.1)
    batch = int(trial.suggest_discrete_uniform('batch', 60, 80,5))
    weight_decay = trial.suggest_loguniform('weight_decay',0.0006,0.0007)
    momentum = trial.suggest_uniform('momentum', 0.7, 0.71)
    epochs = int(trial.suggest_discrete_uniform('epochs', 50, 100,10))
    # optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    optimizer = "SGD"

    # lr0 = 0.09677
    # batch = 70
    # momentum = 0.70297
    # weight_decay = 0.00067
    # epochs = 1000
    optimizer = "SGD"

    # define the model
    model = YOLO("yolo11n-seg.pt").to('cuda')
    
    home = os.getcwd()
    dataset_name = "maize-uav-crop-disease"
    
    # Initialize YOLOv11 model
    model.tune(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,lr0=lr0
        ,batch=batch
        ,momentum=momentum
        ,weight_decay=weight_decay
        ,epochs = epochs
        ,optimizer=optimizer
        ,project="yolo11n_demo"
        ,name="yolo11n_tuning"
        ,device = "0,1"
        ,patience = 10
        ,save = True
        ,save_period = 5
        ,cache = True

    )

    # Evaluate the model and return a metric (e.g., mAP)
    metrics = model.val(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,device = "0,1"
    )
    
    mAP = metrics.seg.map
    
    # model.save("YOLO11n-seg_trained_tuning.pt")

    return mAP  # Higher is better in this case

study = optuna.create_study(direction='maximize')  # We want to maximize the mAP
study.optimize(objective, n_trials=50)  # Run for 50 trials (or more based on resources)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
