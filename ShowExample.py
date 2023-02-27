from PIL import Image
import xml.etree.ElementTree as ET
from Utils import *
import torch

def outputImageAnnotations(fileName, xmlFile, outputFile):

    image = Image.open(fileName).convert('RGB') #bringing sample in as RGB

    labels = []
    boxes = []

    xmlTree = ET.parse(xmlFile)
    xmlRoot = xmlTree.getroot()

    classes = [None,  "nonfluorescent", "fluorescent", "ambiguous" ]   

    for obj in xmlRoot.findall('object'):

        xmin = int(float(obj.find('bndbox').find('xmin').text))
        xmax = int(float(obj.find('bndbox').find('xmax').text))

        ymin = int(float(obj.find('bndbox').find('ymin').text))
        ymax = int(float(obj.find('bndbox').find('ymax').text))


        if(not (xmin==xmax or ymin==ymax)):
            #exclude spurious boxes with 0 area
            labels.append(classes.index(obj.find('name').text))
            boxes.append([xmin, ymin, xmax, ymax])


    annotations = {}
    annotations["labels"] = torch.as_tensor(labels, dtype = torch.int64)
    annotations["boxes"] = torch.as_tensor(boxes, dtype = torch.float)

    outputAnnotatedImgCV(fileName, annotations, outputFile, tensor=False, bbox=True)

outputImageAnnotations("B2x22_inference.png", "B2x22_inference.xml", "outEx.png")