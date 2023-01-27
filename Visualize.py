from Utils import *
import shutil
import os

aberrantList = [l.strip() for l in open("Inference/YYear_OriginalImagesforEarVision2/AberrantYyearMales_ForDiana.txt", 'r').readlines()]

for ear in aberrantList:
    #shutil.copy2("Inference/YYear_OriginalImagesforEarVision2/"+ear+".png", "Inference/YYear_OriginalImagesforEarVision2/AberrantMales/"+ear+".png")
    #shutil.copy2("Inference/YYear_OriginalImagesforEarVision2/"+ear+".xml", "Inference/YYear_OriginalImagesforEarVision2/AberrantMales/"+ear+".xml")

    os.makedirs("Inference/YYear_OriginalImagesforEarVision2/AberrantMales/PointImages", exist_ok=True)
    outputPointAnnotatedImg("Inference/YYear_OriginalImagesforEarVision2/"+ear+".png", "Inference/YYear_OriginalImagesforEarVision2/"+ear+".xml","Inference/YYear_OriginalImagesforEarVision2/AberrantMales/PointImages/"+ear+"_points.png")

    