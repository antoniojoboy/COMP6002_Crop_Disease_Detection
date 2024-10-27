import optuna
import os
import wandb
from ultralytics import YOLO  

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="yolo11n",
    # Track hyperparameters and run metadata
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def objective(trial):
    # Define hyperparameters based on the trial
    lr0 = trial.suggest_loguniform('learning_rate', 0.0001, 0.11)
    batch = int(trial.suggest_discrete_uniform('batch', 4, 100,4))
    weight_decay = trial.suggest_loguniform('weight_decay',0.0001,0.1)
    momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    # define the model
    model = YOLO("yolo11n-seg.pt").to('cuda')
    
    epochs = 100    
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
        ,project="yolo11n"
        ,name="yolo11n"
        ,device = "0,1"
        # ,patience = 5
        ,save = True
        # ,save_period = 5
        ,cache = True

    )

    # Evaluate the model and return a metric (e.g., mAP)
    metrics = model.val(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,device = "0,1"
    )
    
    mAP = metrics.seg.map
    return mAP  # Higher is better in this case

study = optuna.create_study(direction='maximize')  # We want to maximize the mAP
study.optimize(objective, n_trials=50)  # Run for 50 trials (or more based on resources)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
