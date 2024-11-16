import os 

# os.system("pip install roboflow")
# i have deleted this so im not worried about it
# its just for demo of how it was set up
api_key = "Wp8GYbxNZgLdlrtf1eCB"

from roboflow import Roboflow
rf = Roboflow(api_key=api_key)



# project = rf.workspace("zkamlasi-kamlasi-hj4wj").project("plantvillage-dataset")
# version = project.version(1)
# dataset = version.download("yolov11")
                

project = rf.workspace("galihnugraha").project("maize-disease-20240221")
version = project.version(8)
dataset = version.download("yolov11")


# project = rf.workspace("shoaib-hossain-ut2m8").project("crop-disease-segmentation")
# version = project.version(4)
# dataset = version.download("yolov11")


# project = rf.workspace("leafcv").project("leafcv-3.0")
# version = project.version(10)
# dataset = version.download("yolov11")
                