import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET


class ObjectDetectionDataset(Dataset):

    def __init__(self, rootDirectory):
        print("\n----------------------")
        print("DATASET INITIALIZATION")

        self.rootDirectory = rootDirectory  
        self.imageDirectory = self.rootDirectory + "/Images"
        self.annotationDirectory = self.rootDirectory + "/Annotations"
        print("----------------------")        
        print("Dataset Root Directory: ", self.rootDirectory)
        print("Image Directory: ", self.imageDirectory)
        print("Annotations Directory: ", self.annotationDirectory)

        self.imagePaths = []
        self.annotationPaths = []    

        self.classes = [None, "nonfluorescent", "fluorescent"]    
        self.width = 1920
        self.height = 746

        
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



    def __len__(self):
        return len(self.imagePaths)


    def __getitem__(self, index):
        #print("Getting item: ", self.samplePaths[index],self.maskPaths[index])
        #consider changing something here to suppres PIL warning re: RGBA
        image = Image.open(self.imagePaths[index]).convert('RGB') #bringing sample in as RGB
        imgWidth, imgHeight = image.size
        image = image.resize((self.width, self.height))
        imageTensor = TF.to_tensor(image)


        print(self.imagePaths[index])
        labels = []
        boxes = []


        #very helpful notebook for fruit dataset:
        #https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch/notebook

        #parsing XML annotations
        xmlTree = ET.parse(self.annotationPaths[index])
        xmlRoot = xmlTree.getroot()
     
       
        for obj in xmlRoot.findall('object'):



            xmin = int(obj.find('bndbox').find('xmin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)

            ymin = int(obj.find('bndbox').find('ymin').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            
            #print(xmin, ymin, xmax, ymax)
            

            #xmin = (xmin/imgWidth)*self.width
            #xmax = (xmax/imgWidth)*self.width
            #ymin = (ymin/imgHeight)*self.height
            #ymax = (ymax/imgHeight)*self.height


            #print(xmin, ymin, xmax, ymax)

            if(not (xmin==xmax or ymin==ymax)):
                #exclude spurious boxes with 0 area
                labels.append(self.classes.index(obj.find('name').text))
                boxes.append([xmin, ymin, xmax, ymax])

        #Fast-RCNN expects targets as dictionary
        annotations = {}
        annotations["labels"] = torch.as_tensor(labels, dtype = torch.int64)
        annotations["boxes"] = torch.as_tensor(boxes, dtype = torch.float)

        return imageTensor, annotations
