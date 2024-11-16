import optuna
import os
import wandb
from ultralytics import YOLO  

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="tune_yolo11",
    # Track hyperparameters and run metadata
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def objective(trial):
    # Define hyperparameters based on the trial
    lr0 = trial.suggest_loguniform('learning_rate', 0.009, 0.011)
    batch = int(trial.suggest_discrete_uniform('batch', 50, 100,20))
    weight_decay = trial.suggest_loguniform('weight_decay',0.0004,0.0006)
    momentum = trial.suggest_uniform('momentum', 0.92, 0.94)
    epochs = int(trial.suggest_discrete_uniform('epochs', 50, 100,10))
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    # define the model
    # model = YOLO("yolo11n-seg.pt").to('cuda')
    # model.load("YOLO11n-seg_trained_maize-disease-20240221-8.pt")
    model = YOLO("comare_resutls/yolo11n_0.6435/weights/best.pt").to('cuda')
    
    
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
        ,project="tune_yolo11"
        ,name="tune_yolo11"
        ,device = "0,1"
        ,patience = 5
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