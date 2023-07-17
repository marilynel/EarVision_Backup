'''
EarVision 2.0:
Infer

This script loads a pre-trained model, sets a device to run the inference on (preferrably GPU), iterates over the given 
image dataset, and performs an inference using that model. 

Model may be changed in main() where function Infer is called.

Predictions and metrics may be found in C:/Users/CornEnthusiast/Projects/EarVision/Inference/{dataset}
'''

import torch
#import torchvision
import torchvision.transforms.functional as TF
import torchvision.models.detection as objDet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import os
from PIL import Image
from tqdm import tqdm
#import cv2
import numpy as np
import datetime
from Train import outputAnnotatedImgCV, loadHyperparamFile
from Utils import *
import datetime


def Infer(modelDir, epochStr, dirPath = os.getcwd()):
#def Infer(dirPath = os.getcwd()):

    time = datetime.datetime.now().strftime('%m.%d_%H.%M')
    numImagesHandAnno = 0
    
    print("Running EarVision 2.0 Inference")
    print("----------------------")
    print("FINDING GPU")
    print("----------------------")
    print("Currently running CUDA Version: ", torch.version.cuda)

    device = findDevice()

    print(f"Loading Saved Model: {modelDir}\tEpoch: {epochStr}")

    hyperparameters = loadHyperparamFile(f"SavedModels/{modelDir}/Hyperparameters.txt")
    try:
        model = objDet.fasterrcnn_resnet50_fpn_v2(
            weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            box_detections_per_img=700, 
            rpn_pre_nms_top_n_train = hyperparameters["rpn_pre_nms_top_n_train"], 
            rpn_post_nms_top_n_train = hyperparameters["rpn_post_nms_top_n_train"], 
            rpn_pre_nms_top_n_test = hyperparameters["rpn_pre_nms_top_n_test"],   
            rpn_post_nms_top_n_test = hyperparameters["rpn_post_nms_top_n_test"], 
            rpn_fg_iou_thresh = hyperparameters["rpn_fg_iou_thresh"], 
            trainable_backbone_layers = hyperparameters["trainable_backbone_layers"],  
            rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"], 
            box_nms_tresh = hyperparameters["box_nms_thresh"], 
            box_score_thresh = hyperparameters["box_score_thresh"]
        )
    except:
        model = objDet.fasterrcnn_resnet50_fpn_v2(
            weights = objDet.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            box_detections_per_img=700, 
            rpn_pre_nms_top_n_train = hyperparameters["rpn_pre_nms_top_n_train"], 
            rpn_post_nms_top_n_train = hyperparameters["rpn_post_nms_top_n_train"], 
            rpn_pre_nms_top_n_test = hyperparameters["rpn_pre_nms_top_n_test"],   
            rpn_post_nms_top_n_test = hyperparameters["rpn_post_nms_top_n_test"], 
            rpn_fg_iou_thresh = hyperparameters["rpn_fg_iou_thresh"], 
            trainable_backbone_layers = hyperparameters["trainable_backbone_layers"],  
            rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"]
        )
    
    # Potentially add to hyperparameters:
    # rpn_score_thresh = 0.15
    
    # Weird but unless you do this it defaults to 91 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Give the number of classes, including background class
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3) 
    model.to(device)

    # Load saved model and set to eval (because some layers are set to train upon creation)
    model.load_state_dict(torch.load(f"SavedModels/{modelDir}/EarVisionModel_{epochStr}.pt"))
    model.eval() 

    trainingSet = getTrainingSet(modelDir)
    imagePaths = buildImagePathList(dirPath)

    modelID = modelDir.split("/")[-1]    
    inferenceIdentifier = f"InferenceOutput-{modelID[:8]}-{modelID[9:]}-{epochStr}-{time}" 
    
    
    # runinferencesonmultiplemodels edit
    # outputDirectory = dirPath + "/" + inferenceIdentifier
    outputDirectory = dirPath + "/" + inferenceIdentifier + "_test"
    os.makedirs(outputDirectory, exist_ok = True)

    # needAnnotations directory is for images that have more than a specific threshold of predicted ambiguous kernels
    # TODO: this may be changed later to include images with fewer than 200 kernels; TBD 6/27 mel
    newAnnoDir = outputDirectory + "/needAnnotations"
    os.makedirs(newAnnoDir, exist_ok=True)

    outFile = open(f"{outputDirectory}/{inferenceIdentifier}.csv", "w")
    outFile.write(
        "EarName,TrainingSet,PredictedFluor,PredictedNonFluor,PredictedTotal,PredictedTransmission,AmbiguousKernels," +
        "AmbiguousKernelPercentage,AverageEarScoreFluor,AverageEarScoreNonFluor,AverageEarScoreAll,ActualFluor," + 
        "ActualNonFluor,ActualAmbiguous,ActualTotal,ActualTransmission,FluorKernelDiff,FluorKernelABSDiff," + 
        "NonFluorKernelDiff,NonFluorKernelABSDiff,TotalKernelDiff,TotalKernelABSDiff,TransmissionDiff," + 
        "TranmissionABSDiff,PredtoActTransmissionRatio\n"
    )

    # Inference metrics to be used in comparing inferences
    listTransABSDiff, listPredActTransRatios, listPredAmbigs, listTransDiff = [], [] ,[] ,[]

    for path in tqdm(imagePaths):
        # Bring sample in as RGB
        image = Image.open(path).convert('RGB') 
        imageTensor = TF.to_tensor(image).to(device).unsqueeze(0)
        actualFluor, actualNonFluor, actualAmb, actualTransmission, actualTotal = 0, 0, 0, 0, 0
        xmlAvail = True

        try:
            xmlTree = ET.parse(path.split(".")[0] + ".xml")
        except:
            xmlAvail = False

        if xmlAvail:
            xmlAvail, actualFluor, actualNonFluor, actualAmb, actualTransmission, actualTotal = \
                parseXMLData(xmlAvail, xmlTree)


        with torch.no_grad(): 
            prediction = model(imageTensor)[0]

        #keptBoxes = torchvision.ops.nms(prediction['boxes'], prediction['scores'], 0.2 )
        finalPrediction = prediction

        fileName = path.replace("\\", "/").split("/")[-1]

        # Next three lines create files
        outputAnnotatedImgCV(imageTensor[0], finalPrediction, outputDirectory+"/"+ fileName.split(".")[0] + "_inference.png")
        outputPredictionAsXML(finalPrediction, outputDirectory+"/" + fileName.split(".")[0]+"_inference.xml")
        convertPVOC(outputDirectory+"/" + fileName.split(".")[0]+"_inference.xml", image.size)


        predNonFluor = finalPrediction['labels'].tolist().count(1)
        predFluor = finalPrediction['labels'].tolist().count(2)  
    

        # Nex line creates file AND returns number of ambiguous kernels!
        ambiguousKernelCount = findAmbiguousCalls(imageTensor[0], finalPrediction, outputDirectory+"/"+ fileName.split(".")[0] + "_inference.png")
        listPredAmbigs.append(ambiguousKernelCount)

        earName = fileName.split(".")[0] 
        outFile.write(earName+",")

        inTrainingSet = False
        if earName in trainingSet:
            inTrainingSet = True
        

        predNonFluor -= ambiguousKernelCount
        predFluor -= ambiguousKernelCount 
        predTotal = predFluor + predNonFluor 


        imgsForHandAnnotation = False

        if not inTrainingSet and not xmlAvail and ambiguousKernelCount >= 20:
            imgsForHandAnnotation = True
        elif not inTrainingSet and not xmlAvail and predTotal<= 100:
            imgsForHandAnnotation = True



        if imgsForHandAnnotation:
            numImagesHandAnno += 1
            outputAnnotatedImgCV(imageTensor[0], finalPrediction, newAnnoDir+"/"+ fileName.split(".")[0] + "_inference.png")
            outputPredictionAsXML(finalPrediction, newAnnoDir+"/" + fileName.split(".")[0]+"_inference.xml")
            convertPVOC(newAnnoDir+"/" + fileName.split(".")[0]+"_inference.xml", image.size)
            x = findAmbiguousCalls(imageTensor[0], finalPrediction, newAnnoDir+"/"+ fileName.split(".")[0] + "_inference.png")

        try:
            ambiguousKernelPercentage = round(ambiguousKernelCount/(predFluor + predNonFluor - ambiguousKernelCount)*100, 3)     #take total counted kernels, subtract ambiguous kernel count, and use THAT as total to determine percentage
        except:
            ambiguousKernelPercentage  = "N/A"


        ##predNonFluor -= ambiguousKernelCount
        #predFluor -= ambiguousKernelCount 
        #predTotal = predFluor + predNonFluor 

        try:
            predTransmission =   predFluor /  (predTotal) * 100
        except:
            predTransmission = "N/A"

        scores = finalPrediction['scores']
        labels = finalPrediction['labels']

        fluorScores = [score.item() for ind,score in enumerate(scores)  if labels[ind].item()== 2 ]
        nonFluorScores =[score.item() for ind,score in enumerate(scores) if labels[ind].item()== 1 ] 

        # Confidence in the predictions
        avgEarScoreFluor = round(np.mean(fluorScores), 3)
        avgEarScoreNonFluor = round(np.mean(nonFluorScores), 3)
        avgEarScoreAll = round(torch.mean(scores).item(), 3)

        if(inTrainingSet):
            outFile.write("True,")
        else:
            outFile.write(",")

        #write the predictions to outFile
        outFile.write(
            f"{predFluor},{predNonFluor},{predTotal},{predTransmission},{ambiguousKernelCount}," + 
            f"{ambiguousKernelPercentage},{avgEarScoreFluor},{avgEarScoreNonFluor},{avgEarScoreAll},"
        )      

        if(xmlAvail):
            fluorKernelDiff, fluorKernelABSDiff, nonFluorKernelDiff, nonFluorKernelABSDiff, totalKernelDiff, \
                totalKernelABSDiff,  transmissionDiff, transmissionABSDiff = calculateCountMetrics([predFluor, \
                predNonFluor], [actualFluor, actualNonFluor], actualTotalInclAmbig = actualTotal)
            # Include XML data if it is available
            outFile.write(makeStringXmlData(actualFluor, actualNonFluor, actualAmb, actualTotal, actualTransmission, predTransmission,fluorKernelDiff,fluorKernelABSDiff,nonFluorKernelDiff,nonFluorKernelABSDiff,totalKernelDiff,totalKernelABSDiff,transmissionDiff,transmissionABSDiff))
            #outFile.write(f"{actualFluor},{actualNonFluor},{actualAmb},{actualTotal},{actualTransmission},")

            #fluorKernelDiff, fluorKernelABSDiff, nonFluorKernelDiff, nonFluorKernelABSDiff, totalKernelDiff, \
            #    totalKernelABSDiff,  transmissionDiff, transmissionABSDiff = calculateCountMetrics([predFluor, \
            #    predNonFluor], [actualFluor, actualNonFluor], actualTotalInclAmbig = actualTotal)
            
            # Write the metric comparisons between prediced and actual to outFile            
            #outFile.write(
             #   f"{fluorKernelDiff},{fluorKernelABSDiff},{nonFluorKernelDiff},{nonFluorKernelABSDiff}," + 
             #   f"{totalKernelDiff},{totalKernelABSDiff},{transmissionDiff},{transmissionABSDiff}," + 
            #    f"{predTransmission/actualTransmission}"
            #)

            if not inTrainingSet:
                listTransDiff.append(transmissionDiff)
                listTransABSDiff.append(transmissionABSDiff)
                listPredActTransRatios.append(predTransmission/actualTransmission)

        outFile.write("\n")
        
    outFile.close()

    createInfStatsFile(outputDirectory, modelID, epochStr, inferenceIdentifier, listTransABSDiff, \
                       listPredActTransRatios, numImagesHandAnno, listPredAmbigs, listTransDiff)






def makeStringXmlData(actualFluor, actualNonFluor, actualAmb, actualTotal, actualTransmission, predTransmission,fluorKernelDiff,fluorKernelABSDiff,nonFluorKernelDiff,nonFluorKernelABSDiff,totalKernelDiff,totalKernelABSDiff,transmissionDiff,transmissionABSDiff):
    actualDataStr = f"{actualFluor},{actualNonFluor},{actualAmb},{actualTotal},{actualTransmission},"

    #fluorKernelDiff, fluorKernelABSDiff, nonFluorKernelDiff, nonFluorKernelABSDiff, totalKernelDiff, \
    #    totalKernelABSDiff,  transmissionDiff, transmissionABSDiff = calculateCountMetrics([predFluor, \
    #    predNonFluor], [actualFluor, actualNonFluor], actualTotalInclAmbig = actualTotal)
            
            # Write the metric comparisons between prediced and actual to outFile            
    actualDataStr += f"{fluorKernelDiff},{fluorKernelABSDiff},{nonFluorKernelDiff},{nonFluorKernelABSDiff}," + \
        f"{totalKernelDiff},{totalKernelABSDiff},{transmissionDiff},{transmissionABSDiff}," + \
        f"{predTransmission/actualTransmission}"

    return actualDataStr
  
           


def buildImagePathList(imageDirectory):
    imagePaths = []
    for imgIndex, file in enumerate(sorted(os.listdir(imageDirectory))): 
        if(file.endswith((".png", ".jpg", ".tif"))):
            try:
                imagePath = os.path.join(imageDirectory, file)  
                imagePaths.append(imagePath)

            except Exception as e:
                print(str(e))
                pass
    return imagePaths


def parseXMLData(xmlAvail, xmlTree):
    '''
    Parse XML file for images with annotation data available. Return ground truth kernel counts and transmission data. 
    '''
    markerTypeCounts = [0,0,0]
    actualFluor, actualNonFluor, actualAmb, actualTransmission, actualTotal = 0, 0, 0, 0, 0
    xmlRoot = xmlTree.getroot()
    markerData =  xmlRoot.find('Marker_Data')

    for markerType in markerData.findall("Marker_Type"):
        typeID = int(markerType.find('Type').text)
        if(typeID in [1,2,3]):
            markerCount = len(markerType.findall("Marker"))
            markerTypeCounts[typeID-1] = markerCount

    if sum(markerTypeCounts[0:1]) == 0:
        xmlAvail = False
    if xmlAvail:
        actualFluor = markerTypeCounts[0]
        actualNonFluor = markerTypeCounts[1]
        actualAmb = markerTypeCounts[2]

        actualTransmission = actualFluor / (actualNonFluor+actualFluor) * 100

        # should only include fluro and nonfluor, subtract ambiguous
        # actualTotal = sum(markerTypeCounts)   #this would include the ambiguous kernels in the actual total
        actualTotal = actualFluor + actualNonFluor
    #return xmlAvail, actualFluor, actualNonfluor, actualAmb, actualTransmission, actualTotal
    return xmlAvail, actualFluor, actualNonFluor, actualAmb, actualTransmission, actualTotal


def createInfStatsFile(outputDirectory, modelID, epochStr, inferenceIdentifier, listTransABSDiff, \
                       listPredActTransRatios, numImagesHandAnno, listPredAmbigs, listTransDiff):
    with open(f"{outputDirectory}/InferenceStats-{modelID}-{epochStr}.csv", "w") as statsFile:
        statsFile.write(
            "Inference,Model,Date,NotInTrainingSetAvgTransABSDiff,NotInTrainingSetAvgPredActTransRatio," +
            "NumberImagesForHandAnnotation,AllImagesAvgPredAmbigs,NotInTrainingSetAvgTransDiff\n")
        time = datetime.datetime.now().strftime('%m.%d_%H.%M')
        statsFile.write(
            # Inference Identifier  model ID              time
            f"{inferenceIdentifier},{modelID}_{epochStr},{time}," + 
            # average transmission absolute difference 
            f"{sum(listTransABSDiff)/len(listTransABSDiff)}," + 
            f"{sum(listPredActTransRatios)/len(listPredActTransRatios)},{numImagesHandAnno}," +
            f"{sum(listPredAmbigs)/len(listPredAmbigs)},{sum(listTransDiff)/len(listTransDiff)}\n"
        )


def findDevice():
    # Pointing to our GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU. Device: ", device)
    else:
        device = torch.device("cpu")
        print("Running on CPU. Device: ", device)
    return device


def getTrainingSet(modelDir):
    # Read training set data; metrics are taken on images that were not in the training set.
    trainingSetFile = open(f"SavedModels/{modelDir}/TrainingSet.txt").readlines()
    trainingSet = []
    for l in trainingSetFile:
        trainingSet.append(l.strip().replace('\\', '/').split('/')[-1].split('.')[0])
    return trainingSet


if __name__ == "__main__":
    Infer("03.06.23_12.55PM", "022", "Inference/XML_OutTest")