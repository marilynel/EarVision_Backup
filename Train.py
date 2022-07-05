import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as TF
import torchvision.models.detection as objDet
import math
import matplotlib.pyplot as plt
from PIL import ImageDraw
from PIL import ImageFont
import xml.etree.ElementTree as ET
import time
from Trainer import *

from Dataset import ObjectDetectionDataset



#super helpful: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

def main():
    print("EarVision 2.0")
    datasetFull = ObjectDetectionDataset(rootDirectory = "EarDataset")


    validationPercentage = 0.2
    validationSize = math.floor(len(datasetFull)*validationPercentage)

    trainSet, validationSet = torch.utils.data.random_split(datasetFull,[len(datasetFull)-validationSize, validationSize])

    print("Training Set size: ", len(trainSet))
    print("Validation Set size: ", len(validationSet))

    #do i need to use collate_fn here??
    trainingDataLoader = DataLoader(trainSet, batch_size = 2, shuffle=True, collate_fn = myCollate)
    validationDataLoader = DataLoader(validationSet, batch_size = 2, shuffle=False) #setting shuffle to False so it sees the exact same batches during each validation

    #Some code to output examples from validation set.

    
    for i in range(2):
        validateImgEx, validateAnnotationsEx = validationSet.__getitem__(i)
        outputAnnotatedImg(validateImgEx, validateAnnotationsEx, "datasetValidationExample_"+str(i).zfill(3) + ".png")
    
 
    print("----------------------")
    print("FINDING GPU")
    print("----------------------")
    print("Currently running CUDA Version: ", torch.version.cuda)
    #pointing to our GPU if available
    print("Device Count: ", torch.cuda.device_count())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU. Device: ", device)
    else:
        device = torch.device("cpu")
        print("Running on CPU. Device: ", device)


    model = objDet.fasterrcnn_resnet50_fpn_v2(weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_detections_per_img=300).to(device)
    trainer = Trainer(model, trainingDataLoader, validationDataLoader, device)
    startTime = time.time()
    trainer.train()
    endTime = time.time()

    print("----------------------")
    print("ALL TRAINING COMPLETE")
    print("----------------------")
    print("\nTraining Time:", round((endTime-startTime)/60, 4), "minutes")


    model.eval()


    for i in range(2):
        validateImgEx, validateAnnotationsEx = validationSet.__getitem__(i)
        
        with torch.no_grad():
            prediction = model([validateImgEx.to(device)])[0]

        #again, very helpful: https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch/notebook
        keptBoxes = torchvision.ops.nms(prediction['boxes'], prediction['scores'], 0.2 )
        finalPrediction = prediction

        '''
        finalPrediction['boxes'] = finalPrediction['boxes'][keptBoxes]
        finalPrediction['scores'] = finalPrediction['scores'][keptBoxes]
        finalPrediction['labels'] = finalPrediction['labels'][keptBoxes]
        '''

        outputAnnotatedImg(validateImgEx, finalPrediction, "modelOutput_"+str(i).zfill(3) + ".png")

        print(len(finalPrediction['boxes']))
            

def myCollate(batch):
    #from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
    
    '''
    print("Batch")
    for b in batch:
        print(b)
        print("--")
    print("---")
    print("tuple")
    for t in tuple(zip(*batch)):
        print(t)
        print("--")
    '''

    #still slightly unclear about why this is needed, but it resolves eror re: different sized tensors.
    #is the different size tensors from different numbers of objects in diff images?

    return tuple(zip(*batch))


def outputAnnotatedImg(imageTensor, annotations, name="outputImg.png"):
    img = TF.to_pil_image(imageTensor)
    imDraw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=25)


    labels = annotations["labels"]
    boxes = annotations["boxes"]

    classes = [None, "nonfluorescent", "fluorescent"]
    classColors = [None,(76,0,230),(175,255,0)]

    for ind, label in enumerate(labels):
        #print(label, boxes[ind])
        box = boxes[ind]


        imDraw.text((box[0]+25, box[1]), classes[label], font=font,  fill=classColors[label])
    
        
        #Four points to define the bounding box. 
        '''
        coordinates = [(x1, y1), (x2, y2)]
        (x1, y1)
            *--------------
            |             |
            |             |
            |             |
            |             |
            |             |
            |             |
            --------------*
                        (x2, y2)
        '''
        rect = [(box[0], box[1]), (box[2],box[3])]
        imDraw.rectangle(rect, outline=classColors[label], width=3)
        img.save("OutputImages/"+name)

    
if __name__ == "__main__":
    main()

