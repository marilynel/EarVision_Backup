import torch
from torch.utils.data import DataLoader
import torchmetrics
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




def setHyperParams(hyperParameterInput):
    hyperparameters = {}    #hyperparameterInput

    #Reasonable default hyperparameter values are all coded here for safe keeping. This way training should still proceed even if the parameter config .txt file is missing.
    '''
    Definitions:
        NMS         Non-Maximal Suppression; function to filter out duplicates. NMS takes couples of overlapping boxes having 
                    equal class, and if their overlap is greater than some threshold, only the one with higher probability is 
                    kept. This procedure continues until there are no more boxes with sufficient overlap. Default val = 0.7

        backbone    the network used to compute the features for the model; faster r cnn uses RPN

        RPN         Region Proposal Network

    '''
    defaultHyperparams = {
        "validationPercentage" : 0.2,           # 20% of training data is set aside in each epoch to be used for validation 
        "batchSize" : 16,                       # 16 images are sent to the GPU at a time
        "learningRate" : 0.0005,                # incremental changes in params
        "epochs" : 30,                          # number of rounds

        "rpn_pre_nms_top_n_train" : 3000,       # number of proposals to keep before applying NMS during training
        "rpn_post_nms_top_n_train" : 3000,      # number of proposals to keep after applying NMS during training
        "rpn_pre_nms_top_n_test" : 3000,        # number of proposals to keep before applying NMS during testing 
        "rpn_post_nms_top_n_test" : 3000,       # number of proposals to keep after applying NMS during testing
        "rpn_fg_iou_thresh" : 0.7,              # minimum IoU between the anchor and the GT box so that they can be considered as positive during training of the RPN.
        "rpn_batch_size_per_image" : 512,       # number of anchors that are sampled during training of the RPN for computing the loss
        "min_size" : 800,                       # min size of image to be rescaled before feeding it to the backbone
        "max_size" : 1333,                      # max size of image to be rescaled before feeding it to the backbone
        "trainable_backbone_layers" : 3,        # number of trainable (not frozen) layers starting from final block, values between 0 - 5, with 6 ==  all backbone layers are trainable. default == 3
        "box_nms_thresh" : 0.3,                 # NMS threshold for the prediction head. Used during inference
        "box_score_thresh" : 0.2                # during inference, only return proposals with a classification score greater than box_score_thresh
    }

    for param in defaultHyperparams:
        if param not in hyperparameters or hyperparameters[param] =="":
            hyperparameters[param] = defaultHyperparams[param]

    print("Using Hyperparameters: \n")
    for p in hyperparameters:
        print(str(p) + " : " + str(hyperparameters[p]))

    return hyperparameters


def setTrainingAndValidationSets(datasetFull, hyperparameters):
    # TODO: where is testing set? if it's altogether missing, is it needed for this model? google it when you get in Mel!
    validationSize = math.floor(len(datasetFull)*hyperparameters["validationPercentage"])
    # Training and validation sets are split here. Could cross validation be done here? 
    trainSet, validationSet = torch.utils.data.random_split(datasetFull,[len(datasetFull)-validationSize, validationSize], generator=torch.Generator().manual_seed(42)) #seed????

    # is this line redefining trainSet? is the trainset the same as the original dataset here?
    trainSet.dataset = deepcopy(datasetFull)
    trainSet.dataset.isTrainingSet = True

    print("Training Set size: ", len(trainSet))
    print("Validation Set size: ", len(validationSet))

    return trainSet, validationSet


# Is this even necessary??????
##### TODO MARILYN: change range(whatever) back to range(2)
def createExampleImages(validationSet, model, device, modelDir):
    for i in range(2):
        validateImgEx, validateAnnotationsEx = validationSet.__getitem__(i)
        # TODO 
        #print(f"annotations: {validateAnnotationsEx}")
        outputAnnotatedImgCV(validateImgEx, validateAnnotationsEx, modelDir + "/datasetValidationExample_"+str(i).zfill(3) + ".png")
    
    for i in range(2):
        validateImgEx, validateAnnotationsEx = validationSet.__getitem__(i)
        # TODO
        #print(f"annotations for prediction: {validateAnnotationsEx}")
        with torch.no_grad():
            prediction = model([validateImgEx.to(device)])[0]

        #again, very helpful: https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch/notebook
        keptBoxes = torchvision.ops.nms(prediction['boxes'], prediction['scores'], 0.2 )
        finalPrediction = prediction
    
        #print("finalPrediction:")
        #print(finalPrediction)
        #finalPrediction['boxes'] = finalPrediction['boxes'][keptBoxes]
        #finalPrediction['scores'] = finalPrediction['scores'][keptBoxes]
        #finalPrediction['labels'] = finalPrediction['labels'][keptBoxes]
        
        
        outputAnnotatedImgCV(validateImgEx, finalPrediction, modelDir + "/modelOutput_"+str(i).zfill(3) + ".png")
        # does the following line need to be printed at all?????
        print(len(finalPrediction['boxes']))





def main(hyperparameterInput = {}, searchResultDir = ""):

    hyperparameters = setHyperParams(hyperparameterInput)


    startDateTime = datetime.datetime.now()
    modelDir = "SavedModels/" + searchResultDir + startDateTime.strftime("%m.%d.%y_%I.%M%p")
    os.makedirs(modelDir, exist_ok = True)

    print("EarVision 2.0 \n")

    # NOTE: "EarDataset" is the working directory
    datasetFull = ObjectDetectionDataset(rootDirectory = "EarDataset")
    #datasetFull = ObjectDetectionDataset(rootDirectory = "EarDataset_Subsample")


    # modularize setting the trainSet, validationSet?
    trainSet, validationSet = setTrainingAndValidationSets(datasetFull, hyperparameters)




    #wonder how useful changing the num_workers would be in this instance.
    trainingDataLoader = DataLoader(trainSet, batch_size = hyperparameters["batchSize"], shuffle=True, collate_fn = myCollate)

    #setting shuffle to False so it sees the exact same batches during each validation
    # TODO: would shuffle=True help with cross validation?
    validationDataLoader = DataLoader(validationSet, batch_size = hyperparameters["batchSize"], shuffle=False, collate_fn = myCollate) 
    
        
    #Some code to output examples from validation set.
    os.makedirs("OutputImages", exist_ok=True)
    '''
    for i in range(2):
        validateImgEx, validateAnnotationsEx = validationSet.__getitem__(i)
        outputAnnotatedImgCV(validateImgEx, validateAnnotationsEx, "datasetValidationExample_"+str(i).zfill(3) + ".png")
    '''
    device = findGPU()

    #try changing?  trainable_backbone_layers=3, box_score_thresh = 0.03, box_nms_thresh=0.4, 
    model = objDet.fasterrcnn_resnet50_fpn_v2(weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_detections_per_img=700, 
    rpn_pre_nms_top_n_train = hyperparameters["rpn_pre_nms_top_n_train"],   rpn_post_nms_top_n_train = hyperparameters["rpn_post_nms_top_n_train"],  
    rpn_pre_nms_top_n_test = hyperparameters["rpn_pre_nms_top_n_test"],   rpn_post_nms_top_n_test = hyperparameters["rpn_post_nms_top_n_test"], 
    rpn_fg_iou_thresh = hyperparameters["rpn_fg_iou_thresh"], trainable_backbone_layers = hyperparameters["trainable_backbone_layers"],  
    rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"], box_nms_thresh = hyperparameters["box_nms_thresh"], box_score_thresh = hyperparameters["box_score_thresh"])
    
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

    createExampleImages(validationSet, model, device, modelDir)



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

