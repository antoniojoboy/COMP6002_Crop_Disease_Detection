"""
This is to test the tuned model for maize dataset 
"""

from ultralytics import YOLO
import os

home = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(home)

model_name = "yolo11n-seg" # segmentation
dataset_name = "maize-uav-crop-disease"

# Load a YOLO11n model
model = YOLO(home+"/BEST RESULTS/yolo11/weights/best.pt")
Demo_image = home+"/demo/Demo_image/"
# Run batched inference on a list of images
results = model([
    Demo_image+"demo_image.jpg"
    ])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename=Demo_image+"demo_prediction.jpg")  # save to disk