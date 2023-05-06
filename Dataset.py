import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

import albumentations as A


class ObjectDetectionDataset(Dataset):

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

        imageTensor = TF.to_tensor(image)


        #print(self.imagePaths[index])
        labels = []
        boxes = []


        #very helpful notebook for fruit dataset:
        #https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch/notebook

        #parsing XML annotations
        xmlTree = ET.parse(self.annotationPaths[index])
        xmlRoot = xmlTree.getroot()

        '''
        if(len(xmlRoot.findall('object'))>600):
            print("SO MANY Kernels")
            print(self.imagePaths[index])
            print(len(xmlRoot.findall('object')))
        '''
     
       
        for obj in xmlRoot.findall('object'):

            xmin = int(float(obj.find('bndbox').find('xmin').text))
            xmax = int(float(obj.find('bndbox').find('xmax').text))

            ymin = int(float(obj.find('bndbox').find('ymin').text))
            ymax = int(float(obj.find('bndbox').find('ymax').text))
            


            #exclude spurious boxes with 0 area
            if(not (xmin==xmax or ymin==ymax)):
                #if box is labeled ambiguous, add fluorescent and nonfluorescent box
                if(obj.find('name').text=='ambiguous'):
                    labels.append(1)
                    boxes.append([xmin, ymin, xmax, ymax])     
                    labels.append(2)
                    boxes.append([xmin, ymin, xmax, ymax])               
                else:
                    labels.append(self.classes.index(obj.find('name').text))
                    boxes.append([xmin, ymin, xmax, ymax])


                    '''
                    labels = ["nonfluorescent", "nonfluorescent", "fluorescent", ... ]
                    boxes = [
                        [xmin0, ymin0, xmax0, ymax0],
                        [xmin1, ymin1, xmax1, ymax1],
                        [xmin2, ymin2, xmax2, ymax2],
                        ...
                    ]
                    '''


        if(self.isTrainingSet):
            transform = A.Compose([
                # Flip may be vertical, horizontal, or both
                A.Flip(p=0.5),
                A.MedianBlur (
                    blur_limit=(5,9),
                    always_apply=False,
                    p=0.2
                ),
                # augment color jitter more? change contrast/sat/hue?
                A.ColorJitter (brightness=0.2, contrast=0, saturation=0, hue=0, always_apply=False, p=0.2),
                A.Perspective (
                    scale=(0.05, 0.1),
                    keep_size=True,
                    pad_mode=0,
                    pad_val=0,
                    mask_pad_val=0,
                    fit_output=False,
                    always_apply=False,
                    p=0.2
                )
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['myLabels']))

        else:
            # if not training set, we don't transform?
            transform = A.Compose([A.Blur(p=0.0)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['myLabels']))
            
        numpyIm = np.asarray(image)
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
