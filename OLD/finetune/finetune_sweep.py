import optuna
import os
from ultralytics import YOLO  # Assuming this is the model you're using
# Import the W&B Python Library and log into W&B
import wandb

wandb.login()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def objective(config):
    # # Define hyperparameters based on the trial
    # lr0 = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    # weight_decay = trial.suggest_loguniform('weight_decay',0.01, 0.99)
    # momentum = trial.suggest_uniform('momentum', 0.01, 0.99)
    # # batch_size = trial.suggest_int('batch_size', 8, 10)
    # # confidence_threshold = t rial.suggest_uniform('confidence_threshold', 0.1, 0.9)
    # # iou_threshold = trial.suggest_uniform('iou_threshold', 0.3, 0.9)
    # optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    # define the model
    model = YOLO("yolo11n-seg.pt")
    
    home = os.getcwd()
    dataset_name = "maize-disease-20240221-8"
    
    # Initialize YOLOv11 model
    model.train(
        data=home+"/dataset/"+dataset_name+"/data.yaml"
        ,lr0=config.learning_rate
        ,momentum=config.momentum
        ,weight_decay=config.weight_decay
        ,project="sweep_tuning"
        ,name="tuning"
        ,device = "0,1"
        ,epochs = 1
        ,optimizer = "SGD"
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
    mAP = metrics.seg.map

    return mAP  # Higher is better in this case

def main():
    wandb.init(project="sweep_tuning")
    mAP = objective(wandb.config)
    wandb.log({"mAP": mAP})
    wandb.join()


# 2: Define the search space
sweep_configuration = {
    # "method": "grid",
    # "method": "random",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "mAP"},
    # "parameters": {
    #     "learning_rate": {"values":[0.01,0.02]}
    #     ,"momentum": {"values":[0.91]}
    #     ,"weight_decay": {"values":[0.11,0.12]}
    # }
    #     "learning_rate": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    #     ,"momentum": [0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
    #     ,"weight_decay": [0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18]
    # },
    "parameters": {
        "learning_rate": {"max": 0.088, "min": 0.00088}
        ,"momentum": {"max": 0.99, "min": 0.9}
        ,"weight_decay": {"max": 0.2, "min": 0.1}
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep_tuning")

wandb.agent(sweep_id, function=main)