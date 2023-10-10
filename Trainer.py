import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
import numpy as np
import datetime
import os
from Utils import *

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou



#This Trainer class is what actually handles the training loop, given a model and the dataset

class Trainer():
    def __init__(self, network, trainDataLoader, validationDataLoader, device, hyperparameters, saveDirectory):
        self.device = device
        self.network = network
        self.hyperparameters = hyperparameters
        # May consider different optimizers here as well
        self.optimizer = optim.Adam(network.parameters(), lr = self.hyperparameters["learningRate"])
        #look into trying torch.optim.lr_scheduler.StepLR
        self.trainingLoader = trainDataLoader
        self.validationLoader = validationDataLoader
        self.modelDir = saveDirectory

        self.trainingLog = open(f"{self.modelDir}/TrainingLog.csv", "w+")
        self.trainingLog.write("Epoch,InSampleLoss,OutSampleLoss,OutFluorKernelDiff,OutFluorABSDiff,OutNonFluorKernelDiff," + 
                               "OutNonFluorKernelABSDiff,OutTotalKernelDiff,OutTotalKernelABSDiff,OutTransmissionDiff," + 
                               "OutTransmissionABSDiff,InFluorKernelDiff,InFluorABSDiff,InNonFluorKernelDiff," + 
                               "InNonFluorKernelABSDiff,InTotalKernelDiff,InTotalKernelABSDiff,InTransmissionDiff," + 
                               "InTransmissionABSDiff,f1Fluor,f1NonFluor,tMAPgeneral,tMAP50,tMAP75,vMAPgeneral,vMAP50,vMAP75\n")


    def forwardPass(self, images, annotations, train=False, trackMetrics = True):
        if train:
            #zero the gradient
            self.network.zero_grad()

        modelOutput = self.network(images, annotations)

        return modelOutput
    

    def print_metrics(self, avgFluorKernelMiscount, avgFluorABSDiff, avgNonFluorKernelMiscount, avgNonFluorABSDiff, avgTotalKernelMiscount, avgTotalABSDiff, avgTransmissionDiff, avgTransmissionABSDiff,f1FluorAvg, f1NonFluorAvg):
        print(f"-Avg Fluor Kernel Miscount: {avgFluorKernelMiscount}")
        print(f"-Avg Fluor Kernel Miscount (Absolute Value): {avgFluorABSDiff}")

        print(f"-Avg NonFluor Kernel Miscount: {avgNonFluorKernelMiscount}")
        print(f"-Avg NonFluor Kernal Miscount (Absolute Value):{avgNonFluorABSDiff}")

        print(f"-Avg Total Kernel Miscount: {avgTotalKernelMiscount}")
        print(f"-Avg Total Kernal Miscount (Absolute Value): {avgTotalABSDiff}")

        print(f"-Avg Transmission Diff: {avgTransmissionDiff}")
        print(f"-Avg Transmission Diff (Absolute Value): {avgTransmissionABSDiff}")

        print(f"-Avg Fluor F1 Score: {f1FluorAvg}")
        print(f"-Avg NonFluor F1 Score: {f1NonFluorAvg}")

        print('----')


    def splitBoxesByLabel(self, preds, target):
        '''
        0 = None (background)
        1 = nonfluorescent
        2 = fluorescent
        '''
        fluorPredBoxes, nonFluorPredBoxes = [], []

        #print(f"num preds = {len(preds['boxes'])} ({len(preds['labels'])})")
        #print(f"num target = {len(target['boxes'])} ({len(target['labels'])})")

        for i in range(0, len(preds["boxes"])):
            if preds["labels"][i] == 1:
                nonFluorPredBoxes.append(torch.as_tensor(preds["boxes"][i]))
            elif preds["labels"][i] == 2:
                fluorPredBoxes.append(torch.as_tensor(preds["boxes"][i]))
            else:
                print(f"ERROR -- Weird label: {preds['labels'][i]}")

        fluorActBoxes, nonFluorActBoxes = [], []
        for i in range(0, len(target["boxes"])):
            if target["labels"][i] == 1:
                nonFluorActBoxes.append(torch.as_tensor(target["boxes"][i]))
            elif target["labels"][i] == 2:
                fluorActBoxes.append(torch.as_tensor(target["boxes"][i]))
            else:
                print(f"ERROR -- Weird label: {target['labels'][i]}")

        fab = torch.empty(0)
        #fab = torch.empty(1,4)#.fill_(0.)
        nfab = torch.empty(0)
        #nfab = torch.empty(1,4)#.fill_(0.)

        if fluorActBoxes:
            fab = torch.stack(fluorActBoxes)
        if nonFluorActBoxes:
            nfab = torch.stack(nonFluorActBoxes)
        
        fpb = torch.empty(0)
        nfpb = torch.empty(0)

        if fluorPredBoxes:
            fpb = torch.stack(fluorPredBoxes)
        if nonFluorPredBoxes:
            nfpb = torch.stack(nonFluorPredBoxes)
        
        return fpb, fab, nfpb, nfab


    def compPredsVsAnnotation(self, ious, numActualBoxes):                        
        truePos, falsePos, falseNeg = 0, 0, 0                
        boxIdx = [i for i in range(0, numActualBoxes)]
        foundBoxes = []

        if ious is not None:
            for i in range(0, len(ious)):
                match = False
                for j in range(0, len(ious[i])):
                    if ious[i][j] >= 0.7:
                        # TODO: what if there are two boxes with >= 0.7 overlap??? is that possible???
                        match = True
                        foundBoxes.append(boxIdx[j])
                        truePos += 1
                if not match:
                    falsePos += 1

        for box in boxIdx:
            if box not in foundBoxes:
                falseNeg += 1

        boxIdx.clear()
        foundBoxes.clear()

        return truePos, falsePos, falseNeg
        

    def train(self):
        # Enable the gradient
        torch.set_grad_enabled(True)
        epochs = self.hyperparameters["epochs"]

        trainingMAP = MeanAveragePrecision()
        validationMAP = MeanAveragePrecision()
        #trainingMAP = MeanAveragePrecision(iou_thresholds=[0.5,0.75])
        #validationMAP = MeanAveragePrecision(iou_thresholds=[0.5,0.75])

        for e in range(epochs):
            print("------------------")
            print(f"COMMENCING EPOCH: {e+1} / {epochs}")
            print("------------------")

            trainingLossTotal = 0

            # In sample differences in kernel counts, by type, across all in sample images (sum)
            inTotalFluorKernelDiff = 0           
            inTotalNonFluorKernelDiff = 0       
            inTotalTotalKernelDiff = 0          

            inTotalTransmissionDiff = 0
            inTotalFluorKernelABSDiff = 0
            inTotalNonFluorKernelABSDiff = 0
            inTotalTotalABSDiff = 0
            inTotalTransmissionABSDiff = 0

            for batch in tqdm(self.trainingLoader):
                self.network.train()
                images, annotations = batch
                # Weirdly, the F-RCNN model requires inputs in form of list
                # So we gotta turn the batch tensors into lists, and also send each separate item to GPU
                images = list(image.to(self.device) for image in images)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]

                # consider :  with torch.cuda.amp.autocast(enabled=scaler is not None):?
                lossDict = self.forwardPass(images, annotations, train=True)
                # Loss reduction  -- mean vs. sum????
                lossSum = sum(loss for loss in lossDict.values())
                lossMean = lossSum/self.trainingLoader.batch_size
                trainingLossTotal += lossSum
                # trainingLossTotal += lossMean
                lossSum.backward()
                # lossMean.backward()
                self.optimizer.step()
                self.network.eval()
                
                with torch.no_grad():
                    # The reason we run this through a second time just to get the outputs is due to a PyTorch F-CNN 
                    # implementation quirk -- needs to be in eval() mode to get actual outputs.
                    finalPredictions = self.network(images) 

                    trainingMAP.update(finalPredictions, annotations)

                    for i, p in enumerate(finalPredictions):
                        # Count total predicted fluor, predicted nonfluor, actual fluor, and actual nonfluor for an image.
                        # Calculate metrics. 
                        predictedFluorCnt = p['labels'].tolist().count(2)
                        predictedNonFluorCnt = p['labels'].tolist().count(1)
                        actualFluorCnt = annotations[i]['labels'].tolist().count(2)
                        actualNonFluorCnt = annotations[i]['labels'].tolist().count(1)
                        
                        fluorKernelDiff, fluorKernelABSDiff, nonFluorKernelDiff, nonFluorKernelABSDiff, totalKernelDiff, \
                            totalKernelABSDiff, transmissionDiff, transmissionABSDiff = calculateCountMetrics([
                            predictedFluorCnt, predictedNonFluorCnt], [actualFluorCnt, actualNonFluorCnt])

                        inTotalFluorKernelDiff += fluorKernelDiff
                        inTotalFluorKernelABSDiff += fluorKernelABSDiff 

                        inTotalNonFluorKernelDiff += nonFluorKernelDiff
                        inTotalNonFluorKernelABSDiff += nonFluorKernelABSDiff

                        inTotalTotalKernelDiff += totalKernelDiff
                        inTotalTotalABSDiff += totalKernelABSDiff

                        inTotalTransmissionDiff += transmissionDiff
                        inTotalTransmissionABSDiff += transmissionABSDiff

            # Find average insample (training set) metrics across all images in training set
            inAvgFluorKernelMiscount = inTotalFluorKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgFluorABSDiff = inTotalFluorKernelABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgNonFluorKernelMiscount = inTotalNonFluorKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgNonFluorABSDiff = inTotalNonFluorKernelABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgTotalKernelMiscount = inTotalTotalKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgTotalABSDiff = inTotalTotalABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgTransmissionDiff = inTotalTransmissionDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgTransmissionABSDiff = inTotalTransmissionABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            avgTrainingLoss = (trainingLossTotal / len(self.trainingLoader))

            print("computing tMAP....")
            tMAP = trainingMAP.compute()
            trainingMAP.reset()
            print("Training loss:", avgTrainingLoss )
            print("Traning mAP: ", tMAP)
            print("Training Errors: ")

            self.print_metrics(
                inAvgFluorKernelMiscount, inAvgFluorABSDiff, inAvgNonFluorKernelMiscount, inAvgNonFluorABSDiff, 
                inAvgTotalKernelMiscount, inAvgTotalABSDiff, inAvgTransmissionDiff, inAvgTransmissionABSDiff, "n/a", "n/a"
            )

            print("VALIDATING")
            # For quirky reasons, the network needs to be in .train() mode to get the validation loss... silly
            # But it should be kosher since there are no dropout layers and the BatchNorms are frozen
            
            f1FluorValues, f1NonFluorValues = [], []

            with torch.no_grad():    
                totalFluorKernelDiff = 0
                totalNonFluorKernelDiff = 0
                totalTotalKernelDiff = 0
                totalTransmissionDiff = 0
                validationLossTotal = 0
                totalFluorKernelABSDiff = 0
                totalNonFluorKernelABSDiff = 0
                totalTotalABSDiff = 0
                totalTransmissionABSDiff = 0

                for batch in tqdm(self.validationLoader):
                    images, annotations = batch

                    images = list(image.to(self.device) for image in images)
                    annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]

                    self.network.train() # Just b/c of a technicality
                    lossDict = self.forwardPass(images, annotations, train = False)
                    lossSum = sum(loss for loss in lossDict.values())
                    lossMean = lossSum/self.trainingLoader.batch_size

                    validationLossTotal += lossSum
                    # validationLossTotal += lossMean
             
                    # Now have to switch to eval to get actual predictions instead of losses. And the same batch has to 
                    # run through model twice if you want both loss and accuracy metrics -__- 
                    # Seems like a big problem with the model source code. Not sure why they wrote it that way.
                    self.network.eval()
                    predictions = self.network(images)

                    # F1 Calculations begin here
                    finalPredictions = predictions

                    # labels = [None, "nonfluorescent",  "fluorescent"]
                    for k in range(1, len(finalPredictions)):           
                        # "k" is an image in validation set. Iterate through predictions in validation set images and 
                        # calculate F1 scores for each image; append to list.

                        if finalPredictions[k]["boxes"].size(dim = 0) != 0 and annotations[k]["boxes"].size()[0] != 0:
                            fluorPredBoxes, fluorActBoxes, nonFluorPredBoxes, nonFluorActBoxes = \
                                self.splitBoxesByLabel(finalPredictions[k], annotations[k])
                            iousFluor, iousNonFluor = None, None
                            
                            '''
                            TODO: 
                            This calculation is currently not accounting for situations where:
                                ->  there are no predictions
                                ->  there are no actual boxes
                            figure out how to handle them!! (currenlty appended to dataset as 0s)
                            '''

                            # These conditionals are necessary as an image may only have fluor or nonfluor data; above 
                            # conditional will not catch this
                            if fluorPredBoxes.size()[0] != 0 and fluorActBoxes.size()[0] != 0:
                                iousFluor = box_iou(fluorPredBoxes, fluorActBoxes)

                            if nonFluorPredBoxes.size()[0] != 0 and nonFluorActBoxes.size()[0] != 0:
                                iousNonFluor = box_iou(nonFluorPredBoxes, nonFluorActBoxes)   

                            if iousFluor is not None:
                                truePosFluor, falsePosFluor, falseNegFluor = self.compPredsVsAnnotation(iousFluor, len(fluorActBoxes))
                                f1FluorValues.append((2 * truePosFluor) / ((2 * truePosFluor) + falsePosFluor + falseNegFluor))

                            if iousNonFluor is not None:
                                truePosNonFluor, falsePosNonFluor, falseNegNonFluor = self.compPredsVsAnnotation(iousNonFluor, len(nonFluorActBoxes))
                                f1NonFluorValues.append((2 * truePosNonFluor) / ((2 * truePosNonFluor) + falsePosNonFluor + falseNegNonFluor))
    
                        else:
                            f1FluorValues.append(0)
                            f1NonFluorValues.append(0)

                    validationMAP.update(finalPredictions, annotations)

                    for i, p in enumerate(finalPredictions):
                        
                        predictedFluorCnt = p['labels'].tolist().count(2) 
                        predictedNonFluorCnt = p['labels'].tolist().count(1)

                        actualFluorCnt = annotations[i]['labels'].tolist().count(2)
                        actualNonFluorCnt = annotations[i]['labels'].tolist().count(1)
                        
                        fluorKernelDiff, fluorKernelABSDiff, nonFluorKernelDiff, nonFluorKernelABSDiff, totalKernelDiff, \
                            totalKernelABSDiff, transmissionDiff, transmissionABSDiff = calculateCountMetrics([ 
                            predictedFluorCnt, predictedNonFluorCnt], [actualFluorCnt, actualNonFluorCnt])

                        totalFluorKernelDiff += fluorKernelDiff
                        totalFluorKernelABSDiff += fluorKernelABSDiff

                        totalNonFluorKernelDiff += nonFluorKernelDiff
                        totalNonFluorKernelABSDiff += nonFluorKernelABSDiff

                        totalTotalKernelDiff += totalKernelDiff
                        totalTotalABSDiff += totalKernelABSDiff

                        totalTransmissionDiff += transmissionDiff
                        totalTransmissionABSDiff += transmissionABSDiff

            earImageCount =( len(self.validationLoader) * self.validationLoader.batch_size)

            avgFluorKernelMiscount = totalFluorKernelDiff / earImageCount
            avgFluorABSDiff = totalFluorKernelABSDiff / earImageCount

            avgNonFluorKernelMiscount = totalNonFluorKernelDiff / earImageCount
            avgNonFluorABSDiff = totalNonFluorKernelABSDiff / earImageCount

            avgTotalKernelMiscount = totalTotalKernelDiff / earImageCount
            avgTotalABSDiff = totalTotalABSDiff / earImageCount

            avgTransmissionDiff = totalTransmissionDiff / earImageCount
            avgTransmissionABSDiff = totalTransmissionABSDiff / earImageCount

            avgValidationLoss = validationLossTotal /  (len(self.validationLoader) )  

            valMAP = validationMAP.compute()
            validationMAP.reset()
            print("Validation mAP: " , valMAP)

            '''
            TODO: 
            -   decide how to evaluate and note down 0/0 (nan) situations
            '''
            f1FluorAvg, f1NonFluorAvg = 0, 0
            try:
                f1FluorAvg = sum(f1FluorValues) / len(f1FluorValues)
            except:
                f1FluorAvg = 0 

            try:
                f1NonFluorAvg = sum(f1NonFluorValues) / len(f1NonFluorValues)
            except:
                f1NonFluorAvg = 0


            print("Validation Loss: " , avgValidationLoss)
            print("Validation Errors: ")
            
            self.print_metrics(
                avgFluorKernelMiscount, avgFluorABSDiff, avgNonFluorKernelMiscount, avgNonFluorABSDiff, 
                avgTotalKernelMiscount, avgTotalABSDiff, avgTransmissionDiff, avgTransmissionABSDiff, f1FluorAvg, 
                f1NonFluorAvg
            )
                
            self.trainingLog.writelines([f"{e+1},", f"{avgTrainingLoss.item()},", f"{avgValidationLoss.item()},"])
            self.trainingLog.write(f"{avgFluorKernelMiscount},{avgFluorABSDiff},{avgNonFluorKernelMiscount}," + 
                                   f"{avgNonFluorABSDiff},{avgTotalKernelMiscount},{avgTotalABSDiff},{avgTransmissionDiff}," + 
                                   f"{avgTransmissionABSDiff},{inAvgFluorKernelMiscount},{inAvgFluorABSDiff}," + 
                                   f"{inAvgNonFluorKernelMiscount},{inAvgNonFluorABSDiff},{inAvgTotalKernelMiscount}," + 
                                   f"{inAvgTotalABSDiff},{inAvgTransmissionDiff},{inAvgTransmissionABSDiff}," +
                                   f"{f1FluorAvg},{f1NonFluorAvg},{tMAP['map']},{tMAP['map_50']},{tMAP['map_75']}," +
                                   f"{valMAP['map']},{valMAP['map_50']},{valMAP['map_75']}\n")

            torch.save(self.network.state_dict(), f"{self.modelDir}/EarVisionModel_{str(e+1).zfill(3)}.pt")
            print(f"Saved EarVisionModel_{str(e+1).zfill(3)}.pt")

            print(f"\n~EPOCH {e+1} TRAINING COMPLETE~ \n")

        self.trainingLog.close()

