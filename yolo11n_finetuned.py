import optuna
import os
import wandb
from ultralytics import YOLO
from optuna.samplers import TPESampler

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="yolo11n_finetuned",
    # Track hyperparameters and run metadata
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def objective(trial):
 
    # Define hyperparameters based on the trial
    lr0 = trial.suggest_loguniform('learning_rate', 0.0095, 0.0096)
    # batch = int(trial.suggest_discrete_uniform('batch', 4, 100,4))
    weight_decay = trial.suggest_loguniform('weight_decay',0.0004,0.00045)
    momentum = trial.suggest_uniform('momentum', 0.935, 0.94)
    # optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    # iterations = int(trial.suggest_discrete_uniform('batch', 100, 1000,100))

    
    # lr0 = 0.01    
    # epochs = 100  
    iou = 0.7
    # weight_decay = 0.0005
    # momentum = 0.9375
    batch = 64
    iterations = 100000
    optimizer = "auto" 
    home = os.getcwd()
    dataset_name = "maize-uav-crop-disease" 

    # define the model
    model = YOLO("yolo11n-seg_trained_maize-uav-crop-disease.pt").to('cuda')
    model = YOLO("yolo11n-seg_trained_hyper_parameters.pt").to('cuda')
    
    # Tune YOLOv11 model
    model.tune(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,lr0=lr0
        # ,iou=iou
        ,batch=batch
        ,momentum=momentum
        ,weight_decay=weight_decay
        ,optimizer=optimizer
        # ,epochs = epochs
        ,iterations=iterations
        ,project="yolo11n_finetuned"
        ,name="yolo11n"
        ,device = "0,1"
        # ,patience = 5
        # ,save = True
        # ,save_period = 5
        # ,cache = True

    )

    # Evaluate the model and return a metric (e.g., mAP)
    metrics = model.val(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,device = "0,1"
    )
    
    mAP = metrics.seg.map
    return mAP  # Higher is better in this case


# sampler = TPESampler(seed=10)

# study = optuna.create_study(
#     pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
#     sampler=sampler
# )
# study.optimize(objective, n_trials=1000)

study = optuna.create_study(sampler=TPESampler(),direction='maximize')  # We want to maximize the mAP
study.optimize(objective, n_trials=50)  # Run for 50 trials (or more based on resources)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
