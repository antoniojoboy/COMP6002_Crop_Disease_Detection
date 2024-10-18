import os 

# os.system("pip install roboflow")

from roboflow import Roboflow
rf = Roboflow(api_key="Wp8GYbxNZgLdlrtf1eCB")



project = rf.workspace("zkamlasi-kamlasi-hj4wj").project("plantvillage-dataset")
version = project.version(1)
dataset = version.download("yolov11")
                

project = rf.workspace("galihnugraha").project("maize-disease-20240221")
version = project.version(8)
dataset = version.download("yolov11")
                