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
import datetime


def Infer(dirPath = os.getcwd()):
    time = datetime.datetime.now().strftime('%m.%d_%H.%M')
    numImagesHandAnno = 0
    
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

    # Change if needed, options below in function declaration
    modelDir, epochStr = pickModel("pref")
    

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
    rpn_batch_size_per_image = hyperparameters["rpn_batch_size_per_image"], box_nms_thresh = 0.3, box_score_thresh = 0.15)
    
    
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
    
    inferenceIdentifier = "InferenceOutput-" + modelDir[:8] + "-" + modelDir[9:] + "-" + epochStr + "-" + time
    outputDirectory = imageDirectory + "/" + inferenceIdentifier

    os.makedirs(outputDirectory, exist_ok = True)

    newAnnoDir = outputDirectory + "/needAnnotations"
    os.makedirs(newAnnoDir, exist_ok=True)

    outFile = open(outputDirectory + "/" + inferenceIdentifier + ".csv", "w")
    outFile.write(
        "EarName,TrainingSet,PredictedFluor,PredictedNonFluor,PredictedTotal,PredictedTransmission,AmbiguousKernels," +
        "AmbiguousKernelPercentage,AverageEarScoreFluor,AverageEarScoreNonFluor,AverageEarScoreAll,ActualFluor," + 
        "ActualNonFluor,ActualAmbiguous,ActualTotal,ActualTransmission,FluorKernelDiff,FluorKernelABSDiff," + 
        "NonFluorKernelDiff,NonFluorKernelABSDiff,TotalKernelDiff,TotalKernelABSDiff,TransmissionDiff," + 
        "TranmissionABSDiff,PredtoActTransmissionRatio\n"
    )

    listTransABSDiff = []
    listPredActTransRatios = []
    listPredAmbigs = []
    listTransDiff = []

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
            actualAmb = markerTypeCounts[2]

            actualTransmission = actualFluor / (actualNonFluor+actualFluor) * 100

            # should only include fluro and nonfluor, subtract ambiguous
            #actualTotal = sum(markerTypeCounts)   #this would include the ambiguous kernels in the actual total
            actualTotal = actualFluor + actualNonFluor
    

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

        # inTrainingSet == False                if True?    no
        # xmlAvail == False                     if true?    no
        # ambiguousKernelCount >= 20            if true?    yes
        
        imgsForHandAnnotation = False
        if not inTrainingSet and not xmlAvail and ambiguousKernelCount >= 20:
            imgsForHandAnnotation = True


        # TODO: add unedited image to handAnno folder 
        # check parameters into unmarkedImg() as well as usage internal to function

        if imgsForHandAnnotation:
            numImagesHandAnno += 1
            #unmarkedImg(path, newAnnoDir+"/"+ fileName.split(".")[0] + "_original.png")
            outputAnnotatedImgCV(imageTensor[0], finalPrediction, newAnnoDir+"/"+ fileName.split(".")[0] + "_inference.png")
            outputPredictionAsXML(finalPrediction, newAnnoDir+"/" + fileName.split(".")[0]+"_inference.xml")
            convertPVOC(newAnnoDir+"/" + fileName.split(".")[0]+"_inference.xml", image.size)
            x = findAmbiguousCalls(imageTensor[0], finalPrediction, newAnnoDir+"/"+ fileName.split(".")[0] + "_inference.png")

        try:
            ambiguousKernelPercentage = round(ambiguousKernelCount/(predFluor + predNonFluor - ambiguousKernelCount)*100, 3)     #take total counted kernels, subtract ambiguous kernel count, and use THAT as total to determine percentage
        except:
            ambiguousKernelPercentage  = "N/A"


        predNonFluor -= ambiguousKernelCount
        predFluor -= ambiguousKernelCount 
        predTotal = predFluor + predNonFluor 

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
        
        # copy and pasted above for subfolder sorting -- will this work??
        #earName = fileName.split(".")[0] 
        #outFile.write(earName+",")

        

        if(inTrainingSet):
            outFile.write("True,")
        else:
            outFile.write(",")

        #write the predictions to outFile
        outFile.write(str(predFluor) + "," + str(predNonFluor) + "," + str(predTotal) + "," + str(predTransmission) + ",")  
        outFile.write(str(ambiguousKernelCount)+"," + str(ambiguousKernelPercentage)+","+ str(avgEarScoreFluor)+","+str(avgEarScoreNonFluor)+","+str(avgEarScoreAll)+",")      

        if(xmlAvail):
            #write to actual values to outFile
            outFile.write(f"{actualFluor},{actualNonFluor},{actualAmb},{actualTotal},{actualTransmission},")

            fluorKernelDiff, fluorKernelABSDiff, nonFluorKernelDiff, nonFluorKernelABSDiff, totalKernelDiff, \
                totalKernelABSDiff,  transmissionDiff, transmissionABSDiff = calculateCountMetrics([predFluor, \
                predNonFluor], [actualFluor, actualNonFluor], actualTotalInclAmbig = actualTotal)
            
            #write the metric comparisons between prediced and actual to outFile            
            outFile.write(
                f"{fluorKernelDiff},{fluorKernelABSDiff},{nonFluorKernelDiff},{nonFluorKernelABSDiff}," + 
                f"{totalKernelDiff},{totalKernelABSDiff},{transmissionDiff},{transmissionABSDiff}," + 
                f"{predTransmission/actualTransmission}"
            )

            listTransDiff.append(transmissionDiff)
            if not inTrainingSet:
                listTransABSDiff.append(transmissionABSDiff)
                listPredActTransRatios.append(predTransmission/actualTransmission)

        outFile.write("\n")
        

    outFile.write("Model," + modelDir + ",Epoch," + epochStr)
    outFile.close()

    with open(outputDirectory+ "/InferenceStats-" + modelDir + "-" + epochStr + ".csv", "w") as statsFile:
        statsFile.write(
            "Inference,Model,Date,NotInTrainingSetAvgTransABSDiff,NotInTrainingSetAvgPredActTransRatio," +
            "NumberImagesForHandAnnotation,AllImagesAvgPredAmbigs,AllAnnoImagesAvgTransDiff\n")
        statsFile.write(
            # Inference Identifier
            f"{inferenceIdentifier},{modelDir}_{epochStr},{time}," + 
            f"{sum(listTransABSDiff)/len(listTransABSDiff)}," + 
            f"{sum(listPredActTransRatios)/len(listPredActTransRatios)},{numImagesHandAnno}," +
            f"{sum(listPredAmbigs)/len(listPredAmbigs)},{sum(listTransDiff)/len(listTransDiff)}\n"
        )


def pickModel(id):
    '''return modelDir, epochStr'''
    if id == "oldAug":
        return "08.18.22_07.13PM", "021"
    elif id == "feb10":
        return "02.10.23_07.04PM", "028"
    elif id == "feb21":
        return "02.21.23_03.39PM", "019"
    elif id == "feb24":                     # new model with augmentations
        return "02.24.23_03.23PM", "023"
    elif id == "feb2702":
        return "02.27.23_02.06PM", "023"
    elif id == "feb2706":
        return "02.27.23_06.02PM", "024"
    elif id == "pref" or id == "march06":   # this is the model & epoch John is currently using for the Maize meeting
        return "03.06.23_12.55PM", "022"
    elif id == "apr13":
        return "04.13.23_04.13PM", "014"
    elif id == "apr21":
        return "04.21.23_09.46AM", "029"
    elif id == "apr25":
        return "04.25.23_07.46AM", "014"


if __name__ == "__main__":
    Infer("Inference/XML_OutTest")