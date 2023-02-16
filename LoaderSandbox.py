import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models.detection as objDet

from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import numpy as np
from Utils import *

import albumentations as A
import cv2

class SandboxDataset(Dataset):

    def __init__(self, rootDirectory):
        print("\n----------------------")
        print("DATASET INITIALIZATION")

        self.rootDirectory = rootDirectory  
        self.imageDirectory = self.rootDirectory + "/All/Images"
        self.annotationDirectory = self.rootDirectory + "/All/Annotations"
        print("----------------------")        
        print("Dataset Root Directory: ", self.rootDirectory)
        print("Image Directory: ", self.imageDirectory)
        print("Annotations Directory: ", self.annotationDirectory)

        self.imagePaths = []
        self.annotationPaths = []    

        self.classes = [None, "nonfluorescent",  "fluorescent"]    

        
        for imgIndex, file in enumerate(sorted(os.listdir(self.imageDirectory))):  #looping through files in directories
            if(file.endswith((".png", ".jpg", ".tif"))):
                try:
                    imagePath = os.path.join(self.imageDirectory, file)  
                    annotationPath = os.path.join(self.annotationDirectory, os.path.splitext(file)[0].split(".")[0]+".xml")
                    self.imagePaths.append(imagePath)
                    self.annotationPaths.append(annotationPath)
                except Exception as e:
                    print(str(e))
                    pass
        
        self.isTrainingSet = False



    def __len__(self):
        return len(self.imagePaths)


    def __getitem__(self, index):
        #print("Getting item: ", self.samplePaths[index],self.maskPaths[index])
        #consider changing something here to suppres PIL warning re: RGBA
        image = Image.open(self.imagePaths[index]).convert('RGB') #bringing sample in as RGB
        #imgWidth, imgHeight = image.size

        numpyIm = np.asarray(image)
        #imageTensor = TF.to_tensor(image)

        #print(self.imagePaths[index])
        labels = []
        boxes = []

        #parsing XML annotations
        xmlTree = ET.parse(self.annotationPaths[index])
        xmlRoot = xmlTree.getroot()
       
        for obj in xmlRoot.findall('object'):
            xmin = int(float(obj.find('bndbox').find('xmin').text))
            xmax = int(float(obj.find('bndbox').find('xmax').text))

            ymin = int(float(obj.find('bndbox').find('ymin').text))
            ymax = int(float(obj.find('bndbox').find('ymax').text))
            #print(xmin, ymin, xmax, ymax)
            if(not (xmin==xmax or ymin==ymax)):
                #exclude spurious boxes with 0 area
                labels.append(self.classes.index(obj.find('name').text))
                boxes.append([xmin, ymin, xmax, ymax])


        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.0)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['myLabels']))

        transformed = transform(image=numpyIm, bboxes=boxes, myLabels=labels)

        transformedImage = transformed['image']
        transformedBBoxes = transformed['bboxes']
        transformedLabels = transformed['myLabels']
        

        #Faster-RCNN expects targets as dictionary
        annotations = {}
        annotations["labels"] = torch.as_tensor(transformedLabels, dtype = torch.int64)
        annotations["boxes"] = torch.as_tensor(transformedBBoxes, dtype = torch.float)

        imageTensor = TF.to_tensor(transformedImage)

        return imageTensor, annotations


def myCollate(batch):
    return tuple(zip(*batch))


def runSandbox():
    sandboxDir = "./LoaderSandbox/"

    datasetFull = SandboxDataset(rootDirectory = "EarDataset")

    #trainingDataLoader = DataLoader(datasetFull, batch_size = 16, shuffle=True, collate_fn = myCollate)


    for i in range(5):
        exampleItem1 = datasetFull.__getitem__(i)
        exampleItem2 = datasetFull.__getitem__(i)
        exampleItem3 = datasetFull.__getitem__(i)
        exampleItem4 = datasetFull.__getitem__(i)

        outputAnnotatedImgCV(exampleItem1[0], exampleItem1[1], sandboxDir + "example_"+ str(i).zfill(2)+ "_p1.png")
        outputAnnotatedImgCV(exampleItem2[0], exampleItem2[1], sandboxDir + "example_"+ str(i).zfill(2)+ "_p2.png")  
        outputAnnotatedImgCV(exampleItem3[0], exampleItem3[1], sandboxDir + "example_"+ str(i).zfill(2)+ "_p3.png")  
        outputAnnotatedImgCV(exampleItem4[0], exampleItem4[1], sandboxDir + "example_"+ str(i).zfill(2)+ "_p4.png")  

if __name__ == "__main__":
    runSandbox()