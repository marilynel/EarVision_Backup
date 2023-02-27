import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as TF
import torchvision.models.detection as objDet

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
import os
from copy import deepcopy
import numpy as np
from Trainer import *

from Dataset import ObjectDetectionDataset
from Utils import *



def main(hyperparameterInput = {}, searchResultDir = ""):
    print("EarVision 2.0 \n")

    hyperparameters = hyperparameterInput

    #Reasonable default hyperparameter values are all coded here for safe keeping. This way training should still proceed even if the parameter config .txt file is missing.
    defaultHyperparams = {
        "validationPercentage" : 0.2,
        "batchSize" : 16,
        "learningRate" : 0.0005,
        "epochs" : 30, 

        "rpn_pre_nms_top_n_train" : 3000,
        "rpn_post_nms_top_n_train" : 3000,
        "rpn_pre_nms_top_n_test" : 3000,
        "rpn_post_nms_top_n_test" : 3000,
        "rpn_fg_iou_thresh" : 0.7,
        "rpn_batch_size_per_image" : 512,
        "min_size" : 800,
        "max_size" : 1333,
        "trainable_backbone_layers" : 3
    }



    for param in defaultHyperparams:
        if param not in hyperparameters or hyperparameters[param] =="":
            hyperparameters[param] = defaultHyperparams[param]


    print("Using Hyperparameters: \n")
    for p in hyperparameters:
        print(str(p) + " : " + str(hyperparameters[p]))

    datasetFull = ObjectDetectionDataset(rootDirectory = "EarDataset")

    validationSize = math.floor(len(datasetFull)*hyperparameters["validationPercentage"])
    trainSet, validationSet = torch.utils.data.random_split(datasetFull,[len(datasetFull)-validationSize, validationSize], generator=torch.Generator().manual_seed(42)) #seed????

    trainSet.dataset = deepcopy(datasetFull)
    trainSet.dataset.isTrainingSet = True

    print("Training Set size: ", len(trainSet))
    print("Validation Set size: ", len(validationSet))

    #wonder how useful changing the num_workers would be in this instance.
    trainingDataLoader = DataLoader(trainSet, batch_size = hyperparameters["batchSize"], shuffle=True, collate_fn = myCollate)
    validationDataLoader = DataLoader(validationSet, batch_size = hyperparameters["batchSize"], shuffle=False, collate_fn = myCollate) #setting shuffle to False so it sees the exact same batches during each validation

        
    #Some code to output examples from validation set.
    os.makedirs("OutputImages", exist_ok=True)

    for i in range(2):
        validateImgEx, validateAnnotationsEx = validationSet.__getitem__(i)
        outputAnnotatedImgCV(validateImgEx, validateAnnotationsEx, "datasetValidationExample_"+str(i).zfill(3) + ".png")

    device = findGPU()

    #try changing?  trainable_backbone_layers=3, box_score_thresh = 0.03, box_nms_thresh=0.4, 
    '''
    model = objDet.fasterrcnn_resnet50_fpn_v2(weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_detections_per_img=700, 
    rpn_pre_nms_top_n_train = hyperparameters["rpn_pre_nms_top_n_train"],   rpn_post_nms_top_n_train = hyperparameters["rpn_post_nms_top_n_train"],  
    rpn_pre_nms_top_n_test = hyperparameters["rpn_pre_nms_top_n_test"],   rpn_post_nms_top_n_test = hyperparameters["rpn_post_nms_top_n_test"], 
    rpn_fg_iou_thresh = hyperparameters["rpn_fg_iou_thresh"], trainable_backbone_layers = hyperparameters["trainable_backbone_layers"],  
    rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"])
    '''
    model = objDet.fasterrcnn_resnet50_fpn_v2(weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_detections_per_img=700, 
    rpn_pre_nms_top_n_train = hyperparameters["rpn_pre_nms_top_n_train"],   rpn_post_nms_top_n_train = hyperparameters["rpn_post_nms_top_n_train"],  
    rpn_pre_nms_top_n_test = hyperparameters["rpn_pre_nms_top_n_test"],   rpn_post_nms_top_n_test = hyperparameters["rpn_post_nms_top_n_test"], 
    rpn_fg_iou_thresh = hyperparameters["rpn_fg_iou_thresh"], trainable_backbone_layers = hyperparameters["trainable_backbone_layers"],  
    rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"], box_nms_thresh = 0.3, box_score_thresh = 0.15)
    
    #awkward but unless you do this it defaults to 91 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3) #give it number of classes. includes background class as one of the counted classes

    model.to(device)

    #making a time-stamped folder for the training run.
    startDateTime = datetime.datetime.now()
    modelDir = "SavedModels/" + searchResultDir + startDateTime.strftime("%m.%d.%y_%I.%M%p")
    os.makedirs(modelDir, exist_ok = True)

    outputHyperparameterFile(hyperparameters, modelDir)
    outputDataSetList(trainSet, modelDir+"/TrainingSet.txt")
    outputDataSetList(validationSet, modelDir+"/ValidationSet.txt")

    trainer = Trainer(model, trainingDataLoader, validationDataLoader, device, hyperparameters, saveDirectory = modelDir)
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

        outputAnnotatedImgCV(validateImgEx, finalPrediction, "modelOutput_"+str(i).zfill(3) + ".png")

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
    #slightly unclear about why this is needed, but it resolves eror re: different sized tensors.
    #is the different size tensors from different numbers of objects in diff images?

    return tuple(zip(*batch))


def outputDataSetList(dataSet, fileName):

    outFile = open(fileName, "w")
    #the Subset class is so awkward!
    for i in dataSet.indices:
        outFile.write(dataSet.dataset.imagePaths[i] + "\n")
    outFile.close()


def loadHyperparamFile(fileName = "HyperparametersConfig.txt"):
    '''Loads in .txt file with the various hyperparameter values for the training run.'''

    hyperparameters = {}
    with open(fileName, 'r') as f:
        fileLines = f.readlines()
    
    for l in fileLines:
        if l[0]!= "#" and l!="\n":
            parameterEntry = l.strip().split("=")
            key = parameterEntry[0].strip()
            value = parameterEntry[1].lstrip()
            if(value.isnumeric()):    #should convert integer str params to ints
                value = int(value)
            else:
                try:
                    value = float(value)       #should convert float str params to float
                except:
                    value = value              #should grab any str str params as str
            hyperparameters[key] = value

    return hyperparameters

def outputHyperparameterFile(hyperparams, dir):
    outFile = open(dir+"/Hyperparameters.txt", "w")
    for key, value in hyperparams.items():
        outFile.write(str(key)+ " = " + str(value)+"\n")
    outFile.close()

if __name__ == "__main__":
    hyperparameterFile = loadHyperparamFile()
    main(hyperparameterFile)

