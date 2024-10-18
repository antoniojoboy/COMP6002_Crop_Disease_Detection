import optuna
import os
from ultralytics import YOLO  # Assuming this is the model you're using

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def objective(trial):
    # Define hyperparameters based on the trial
    lr0 = trial.suggest_loguniform('learning_rate', 0.08, 0.1)
    weight_decay = trial.suggest_loguniform('weight_decay',0.1, 0.2)
    momentum = trial.suggest_uniform('momentum', 0.9, 0.99)
    # batch_size = trial.suggest_int('batch_size', 8, 10)
    # confidence_threshold = t rial.suggest_uniform('confidence_threshold', 0.1, 0.9)
    # iou_threshold = trial.suggest_uniform('iou_threshold', 0.3, 0.9)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    # define the model
    model = YOLO("yolo11n-seg.pt").to('cuda')
    
    home = os.getcwd()
    dataset_name = "maize-disease-20240221-8"
    
    # Initialize YOLOv11 model
    model.train(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,lr0=lr0
        ,momentum=momentum
        ,weight_decay=weight_decay
        ,project="sweep_hyper_tuning"
        ,name="yolo11n-finetuned"
        ,device = "0,1"
        ,optimizer = "SGD"
        ,epochs = 100
        # ,iterations=1
    )

    # # Train the model (you will need to define this)
    # results_grid = model.train(
    # data="C:/Users/anton/Documents/GitHub/Curtin-Masters_of_Artificial_Intelligence/COMP6002 - Computer Science Project/Custom_Trained/data.yaml"
    # ,device = "cuda"
    # ,epochs = 10
    # # ,iterations= 1
    # )

    # Evaluate the model and return a metric (e.g., mAP)
    metrics = model.val(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,device = "0,1"
    )
    mAP = metrics.box.

    return mAP  # Higher is better in this case






study = optuna.create_study(direction='maximize')  # We want to maximize the mAP
study.optimize(objective, n_trials=50)  # Run for 50 trials (or more based on resources)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
