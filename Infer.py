import torch
import torchvision
import torchvision.transforms.functional as TF
import torchvision.models.detection as objDet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import datetime
from Train import outputAnnotatedImgCV, loadHyperparamFile
from Utils import *


def Infer(dirPath = os.getcwd()):
    print("Running EarVision 2.0 Inference")

    print("----------------------")
    print("FINDING GPU")
    print("----------------------")
    print("Currently running CUDA Version: ", torch.version.cuda)
    #pointing to our GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU. Device: ", device)
    else:
        device = torch.device("cpu")
        print("Running on CPU. Device: ", device)


    modelDir = "08.18.22_07.13PM"
    epochStr = "021"

    #modelDir = "02.10.23_07.04PM"
    #epochStr = "028"

    
    print("Loading Saved Model: ", modelDir,  "    Epoch: ", epochStr)

 
    hyperparameters = loadHyperparamFile("SavedModels/"+modelDir+"/Hyperparameters.txt")

    '''
    model = objDet.fasterrcnn_resnet50_fpn_v2(weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_detections_per_img=700, 
    rpn_pre_nms_top_n_train = hyperparameters["rpn_pre_nms_top_n_train"],   rpn_post_nms_top_n_train = hyperparameters["rpn_post_nms_top_n_train"],  
    rpn_pre_nms_top_n_test = hyperparameters["rpn_pre_nms_top_n_test"],   rpn_post_nms_top_n_test = hyperparameters["rpn_post_nms_top_n_test"], 
    rpn_fg_iou_thresh = hyperparameters["rpn_fg_iou_thresh"], trainable_backbone_layers = hyperparameters["trainable_backbone_layers"],  
    rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"], box_nms_thresh=0.10, box_score_thresh = 0.30, rpn_score_thresh = 0.15)
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
    rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"], box_nms_thresh=0.25)
   '''


    #weird but unless you do this it defaults to 91 classes
    in_features =   model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3) #give it number of classes, including background class
    model.to(device)


    model.load_state_dict(torch.load("SavedModels/"+modelDir+"/EarVisionModel_"+ epochStr  +".pt"))
    model.eval() # set to eval because some layers are set to train upon creation

   
    #just for cases when you run happen to run inference on things that could have been in the training set. Would like to keep track of this
    trainingSetFile = open("SavedModels/"+modelDir+"/TrainingSet.txt").readlines()
    trainingSet = []
    for l in trainingSetFile:
        trainingSet.append(l.strip().replace('\\', '/').split('/')[-1].split('.')[0])

    imageDirectory = dirPath
    imagePaths = []

    for imgIndex, file in enumerate(sorted(os.listdir(imageDirectory))):  #looping through files in directories
        if(file.endswith((".png", ".jpg", ".tif"))):
            try:
                imagePath = os.path.join(imageDirectory, file)  
                imagePaths.append(imagePath)

            except Exception as e:
                print(str(e))
                pass
    
    outputDirectory = imageDirectory + "/InferenceOutput"
    os.makedirs(outputDirectory, exist_ok = True)

    outFile = open(outputDirectory+ "/InferenceCounts.csv", "w")

    outFile.write("EarName,TrainingSet,PredictedFluor,PredictedNonFluor,PredictedTotal,PredictedTransmission,"+
    "AmbiguousKernels,AmbiguousKernelPercentage,AverageEarScoreFluor,AverageEarScoreNonFluor,AverageEarScoreAll," +
    "ActualFluor,ActualNonFluor,ActualTotal,ActualTransmission,"+
    "FluorKernelDiff,FluorPercentageDiff,NonFluorKernelDiff,NonFluorPercentageDiff,TotalKernelDiff,TotalPercentageDiff,TransmissionDiff,TranmissionPercentageDiff" +"\n")

    for path in tqdm(imagePaths):
        #print("working on ", path)
        image = Image.open(path).convert('RGB') #bringing sample in as RGB
        #imgWidth, imgHeight = image.size
        #image = image.resize((self.width, self.height))
        imageTensor = TF.to_tensor(image).to(device).unsqueeze(0)

        markerTypeCounts = [0,0,0]

        xmlAvail = True

        try:
            xmlTree = ET.parse(path.split(".")[0] + ".xml")

        except:
            xmlAvail = False


        if(xmlAvail):
            xmlRoot = xmlTree.getroot()

            markerData =  xmlRoot.find('Marker_Data')

            for markerType in markerData.findall("Marker_Type"):
                typeID = int(markerType.find('Type').text)
                if(typeID in [1,2,3]):
                    markerCount = len(markerType.findall("Marker"))
                    markerTypeCounts[typeID-1] = markerCount

            #print("Marker Type Counts:", markerTypeCounts, "Total: ", sum(markerTypeCounts))

            if sum(markerTypeCounts[0:1])==0:
                xmlAvail = False

        if(xmlAvail):
            actualFluor = markerTypeCounts[0]
            actualNonFluor = markerTypeCounts[1]

            actualTransmission = actualFluor / (actualNonFluor+actualFluor) * 100

            actualTotal = sum(markerTypeCounts)   #this would include the ambiguous kernels in the actual total

    

        #you might as well put the input images into a batch, no? Should process faster...

        with torch.no_grad(): 
            prediction = model(imageTensor)[0]
        #print(prediction)

        #keptBoxes = torchvision.ops.nms(prediction['boxes'], prediction['scores'], 0.2 )
        finalPrediction = prediction

        '''
        finalPrediction['boxes'] = finalPrediction['boxes'][keptBoxes]
        finalPrediction['scores'] = finalPrediction['scores'][keptBoxes]
        finalPrediction['labels'] = finalPrediction['labels'][keptBoxes]
        '''

        fileName = path.replace("\\", "/").split("/")[-1]
   
        outputAnnotatedImgCV(imageTensor[0], finalPrediction, outputDirectory+"/"+ fileName.split(".")[0] + "_inference.png")
        outputPredictionAsXML(finalPrediction, outputDirectory+"/" + fileName.split(".")[0]+"_inference.xml")
        convertPVOC(outputDirectory+"/" + fileName.split(".")[0]+"_inference.xml", image.size)


        predNonFluor = finalPrediction['labels'].tolist().count(1)
        predFluor = finalPrediction['labels'].tolist().count(2)  
    
        predTotal = predFluor + predNonFluor  #currently includes ambiguous
        predTransmission =   predFluor /  (predTotal) * 100

        ambiguousKernelCount = findAmbiguousCalls(imageTensor[0], finalPrediction, outputDirectory+"/"+ fileName.split(".")[0] + "_inference.png")
        ambiguousKernelPercentage = round(ambiguousKernelCount/(predTotal - ambiguousKernelCount)*100, 3)     #take total counted kernels, subtract ambiguous kernel count, and use THAT as total to determine percentage


        scores = finalPrediction['scores']
        labels = finalPrediction['labels']

        fluorScores = [score.item() for ind,score in enumerate(scores)  if labels[ind].item()== 2 ] #update when you retrain!
        nonFluorScores =[score.item() for ind,score in enumerate(scores) if labels[ind].item()== 1 ] 


        avgEarScoreFluor = round(np.mean(fluorScores), 3)
        avgEarScoreNonFluor = round(np.mean(nonFluorScores), 3)
        avgEarScoreAll = round(torch.mean(scores).item(), 3)
        

        earName = fileName.split(".")[0] 
        outFile.write(earName+",")

        inTrainingSet = False
        if earName in trainingSet:
            inTrainingSet = True

        if(inTrainingSet):
            outFile.write("True,")
        else:
            outFile.write(",")

        #write the predictions to outFile
        outFile.write(str(predFluor) + "," + str(predNonFluor) + "," + str(predTotal) + "," + str(predTransmission) + ",")  
        outFile.write(str(ambiguousKernelCount)+"," + str(ambiguousKernelPercentage)+","+ str(avgEarScoreFluor)+","+str(avgEarScoreNonFluor)+","+str(avgEarScoreAll)+",")      

        if(xmlAvail):
            #write to actual values to outFile
            outFile.write(str(actualFluor)+"," + str(actualNonFluor) + "," + str(actualTotal) + ","  + str(actualTransmission) + "," )

            fluorKernelDiff, fluorPercentageDiff, nonFluorKernelDiff, nonFluorPercentageDiff, \
            totalKernelDiff, totalPercentageDiff,  transmissionDiff, transmissionPercentageDiff = calculateCountMetrics([ predFluor, predNonFluor ], [actualFluor, actualNonFluor], actualTotalInclAmbig = actualTotal)
            
            #write the metric comparisons between prediced and actual to outFile

            outFile.write( str(fluorKernelDiff) + "," + str(fluorPercentageDiff) + "," + 
            str(nonFluorKernelDiff) + "," + str(nonFluorPercentageDiff)+","+str(totalKernelDiff)+","+str(totalPercentageDiff)+"," +
            str(transmissionDiff)+","+str(transmissionPercentageDiff))

        outFile.write("\n")

        #print(" ")

    outFile.close()


if __name__ == "__main__":
    Infer("Inference/XML_OutTest")