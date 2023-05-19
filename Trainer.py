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
#from torchmetrics import F1Score
from torchvision.ops import box_iou



#This Trainer class is what actually handles the training loop, given a model and the dataset

class Trainer():
    def __init__(self, network, trainDataLoader, validationDataLoader, device, hyperparameters, saveDirectory):
        self.device = device
        self.network = network
        self.hyperparameters = hyperparameters
        ''' have seen this in some places:    
         params = [p for p in network.parameters() if p.requires_grad], when passing params to optimizer. 
         only calculates trainable parameters '''
        #consider different optimizers here as well
        self.optimizer = optim.Adam(network.parameters(), lr = self.hyperparameters["learningRate"])
        '''note: i have not been using a Learning Rate Scheduler, since I
        have been using Adam. but it might be work trying, either with Adam or
        with a differnt optimizer (like SGD) ''' 
        #look into trying torch.optim.lr_scheduler.StepLR
        self.trainingLoader = trainDataLoader
        self.validationLoader = validationDataLoader
        self.modelDir = saveDirectory


        self.trainingLog = open(self.modelDir + "/TrainingLog.csv", "w+")
        self.trainingLog.write("Epoch,InSampleLoss,OutSampleLoss,OutFluorKernelDiff,OutFluorABSDiff,OutNonFluorKernelDiff," + 
                               "OutNonFluorKernelABSDiff,OutTotalKernelDiff,OutTotalKernelABSDiff,OutTransmissionDiff," + 
                               "OutTransmissionABSDiff,InFluorKernelDiff,InFluorABSDiff,InNonFluorKernelDiff," + 
                               "InNonFluorKernelABSDiff,InTotalKernelDiff,InTotalKernelABSDiff,InTransmissionDiff," + 
                               "InTransmissionABSDiff,f1Fluor,f1NonFluor\n")


    def forwardPass(self, images, annotations, train=False, trackMetrics = True):
        if train:
            #zero the gradient
            self.network.zero_grad()

        modelOutput = self.network(images, annotations)

        return modelOutput
    

    def print_metrics(self, avgFluorKernelMiscount, avgFluorABSDiff, avgNonFluorKernelMiscount, avgNonFluorABSDiff, avgTotalKernelMiscount, avgTotalABSDiff, avgTransmissionDiff, avgTransmissionABSDiff,f1FluorAvg, f1NonFluorAvg):
        print("-Avg Fluor Kernel Miscount: ", avgFluorKernelMiscount)
        print("-Avg Fluor Kernel Miscount (Absolute Value): ", avgFluorABSDiff)

        print("-Avg NonFluor Kernel Miscount: ", avgNonFluorKernelMiscount)
        print("-Avg NonFluor Kernal Miscount (Absolute Value):", avgNonFluorABSDiff)

        print("-Avg Total Kernel Miscount: ",  avgTotalKernelMiscount)
        print("-Avg Total Kernal Miscount (Absolute Value):", avgTotalABSDiff)

        print("-Avg Transmission Diff: ", avgTransmissionDiff)
        print("-Avg Transmission Diff (Absolute Value):", avgTransmissionABSDiff)

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
        #fpb = torch.empty(1,4)#.fill_(0.)
        nfpb = torch.empty(0)
        #nfpb = torch.empty(1,4)#.fill_(0.)

        if fluorPredBoxes:
            fpb = torch.stack(fluorPredBoxes)
        if nonFluorPredBoxes:
            nfpb = torch.stack(nonFluorPredBoxes)
        
        #print(f"fpb: {fpb}")
        #print(f"nfpb: {nfpb}")
        

        return fpb, fab, nfpb, nfab


    def compPredsVsAnnotation(self, ious, numActualBoxes):                    
                        
        truePos, falsePos, falseNeg = 0, 0, 0                

        boxIdx = [i for i in range(0, numActualBoxes)]
        foundBoxes = []

        #print(f"before boxIdx = {boxIdx}")
        #print(f"before foundBoxes = {foundBoxes}")

        if ious is not None:
            #print(f"ious is not none")
            for i in range(0, len(ious)):
                match = False
                #print(f"iou[{i}] = {ious[i]}")
                for j in range(0, len(ious[i])):
                    #print(f"j = {j}")
                    if ious[i][j] >= 0.7:
                        # TODO: what if there are two boxes with >= 0.7 overlap??? is that possible???
                        #print(f"at i = {i}, j = {j}, iou value {ious[i][j]} is greater than or equal to 0.7")
                        match = True
                        #print(f"match is {match}")
                        foundBoxes.append(boxIdx[j])
                        #print(f"foundBoxes contains: {foundBoxes}")
                        truePos += 1
                        #print(f"truePos = {truePos}")
                        # break
                if not match:
                    falsePos += 1
                    #print(f"only print me if match is false (did not find a matching box) for {i}")
                    #print(f"falsePos = {falsePos}")
        #print(f"\n after boxIdx = {boxIdx}")
        #print(f"after foundBoxes = {foundBoxes}")
        for box in boxIdx:
            #print(f"for {box} in boxIdx:")
            if box not in foundBoxes:
                #print(f"\t{box} is not in foundBoxes")
                #print(f"ground truth box {box} is a false negative")
                falseNeg += 1
                #print(f"\tfalseNeg = {falseNeg}")

        boxIdx.clear()
        foundBoxes.clear()

        return truePos, falsePos, falseNeg
        

    def train(self):

        torch.set_grad_enabled(True) #enable the gradient
        epochs = self.hyperparameters["epochs"]
        # NOTE: above line is correct
        #epochs = 10

        # NOTE: if we go back to using MAP, just do vMAP
        # MAP calculation is very time intensive, althought so is F1
        #validationMAP = MeanAveragePrecision(class_metrics = True)
        #validationMAP = MeanAveragePrecision(iou_thresholds=[0.5,0.75],class_metrics=False)

        for e in range(epochs):

            print("------------------")
            print("COMMENCING EPOCH: ", str(e+1)+"/"+str(epochs) )
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
                #weirdly, the F-RCNN model requires inputs in form of list
                #so we gotta turn the batch tensors into lists, and also send each separate item to GPU
                images = list(image.to(self.device) for image in images)
                
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]

                # consider :  with torch.cuda.amp.autocast(enabled=scaler is not None):?
                lossDict = self.forwardPass(images, annotations, train=True)
                
                #loss reduction  -- mean vs. sum????
                lossSum = sum(loss for loss in lossDict.values())

                lossMean = lossSum/self.trainingLoader.batch_size
                
                trainingLossTotal += lossSum
                #trainingLossTotal += lossMean

                lossSum.backward()
                #lossMean.backward()
              
                self.optimizer.step()

                self.network.eval()
                
                with torch.no_grad():
                    #the reason we run this through a second time just to get the outputs is due to a PyTorch F-CNN implementation quirk -- needs to be in eval() mode to get actual outputs.
                    
                    finalPredictions = self.network(images) 
                    for i, p in enumerate(finalPredictions):
                        '''
                        Count total predicted fluor, predicted nonfluor, actual fluor, and actual nonfluor for an image.
                        Calculate metrics. 
                        '''
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


            '''
            Find average insample (training set) metrics across all images in training set
            '''
            inAvgFluorKernelMiscount = inTotalFluorKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgFluorABSDiff = inTotalFluorKernelABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgNonFluorKernelMiscount = inTotalNonFluorKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgNonFluorABSDiff = inTotalNonFluorKernelABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgTotalKernelMiscount = inTotalTotalKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgTotalABSDiff = inTotalTotalABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgTransmissionDiff = inTotalTransmissionDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgTransmissionABSDiff = inTotalTransmissionABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            avgTrainingLoss = (trainingLossTotal / len(self.trainingLoader))

            
            print("Training loss:", avgTrainingLoss )
            print("Training Errors: ")

            self.print_metrics(
                inAvgFluorKernelMiscount, inAvgFluorABSDiff, inAvgNonFluorKernelMiscount, inAvgNonFluorABSDiff, 
                inAvgTotalKernelMiscount, inAvgTotalABSDiff, inAvgTransmissionDiff, inAvgTransmissionABSDiff, "n/a", "n/a"
            )

            print("VALIDATING")
            #because of quirky reasons, the network needs to be in .train() mode to get the validation loss... silly
            #but it should be kosher since there are no dropout layers and the BatchNorms are frozen
            
            f1FluorValues, f1NonFluorValues = [], []

            #self.network.eval()
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

                    self.network.train() #just b/c of a technicality
                    lossDict = self.forwardPass(images, annotations, train = False)
                    #print(lossDict)
                    #print("just evaled validaiton loss")

                    #loss reduction = =mean vs. sum?
                    lossSum = sum(loss for loss in lossDict.values())
                    lossMean = lossSum/self.trainingLoader.batch_size

                    validationLossTotal += lossSum
                    #validationLossTotal += lossMean
             
                    #print("validaiton loss sum:", lossSum)

                    #now have to switch to eval to get actual predictions instead of losses. And the same batch has to run through model twice if you want both loss and accuracy metrics -__- 
                    #seems like a big problem with the model source code. Not sure why they wrote it that way.
                    self.network.eval()
                    predictions = self.network(images)

                    #### START HERE F1 ####
                    finalPredictions = predictions

                    # labels = [None, "nonfluorescent",  "fluorescent"]
                    #            0           1                 2  
                    for k in range(1, len(finalPredictions)):           
                        '''
                        "k" is an image in validation set. Iterate through predictions in validation set images and 
                        calculate F1 scores for each image; append to list.
                        '''                        

                        # Make sure that something exists in annotations and predictions for that image
                        # TODO: how should those situations be handled???
                        if finalPredictions[k]["boxes"].size(dim = 0) != 0 and annotations[k]["boxes"].size()[0] != 0:
                        
                            fluorPredBoxes, fluorActBoxes, nonFluorPredBoxes, nonFluorActBoxes = self.splitBoxesByLabel(finalPredictions[k], annotations[k])

                            iousFluor, iousNonFluor = None, None
                            
                            '''
                            TODO: 
                            currently not accountoing for situations where:
                                ->  there are no predictions
                                ->  there are no actual boxes
                            figure out how to handle them!! (currenlty appended to dataset as 0s)

                            '''

                            # These conditionals are necessary as an image may only have fluor or nonfluor data; above conditional will not catch
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

                    #### END NEW ####


                    #validationMAP.update(finalPredictions, annotations)



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

            # do we need a validationMAP.update()?
            #valMAP = validationMAP.compute()
            #validationMAP.reset()
            #print("Validation mAP: " , valMAP)


            '''
            TODO: MARILYN:
            -   if an F1 score is whacky (no preds, no acts) it is currently going in as a 0 --> should it???
            -   decide how to evaluate and note down 0/0 (nan) situations
            '''

            f1FluorAvg = sum(f1FluorValues) / len(f1FluorValues)
            f1NonFluorAvg = sum(f1NonFluorValues) / len(f1NonFluorValues)
            

            print("Validation Loss: " , avgValidationLoss)
            print("Validation Errors: ")
            
            self.print_metrics(
                avgFluorKernelMiscount, avgFluorABSDiff, avgNonFluorKernelMiscount, avgNonFluorABSDiff, 
                avgTotalKernelMiscount, avgTotalABSDiff, avgTransmissionDiff, avgTransmissionABSDiff, f1FluorAvg, 
                f1NonFluorAvg
            )
                
            self.trainingLog.writelines([str(e+1)+"," , str(avgTrainingLoss.item()) +",", str(avgValidationLoss.item())+","])
            

            self.trainingLog.write(f"{avgFluorKernelMiscount},{avgFluorABSDiff},{avgNonFluorKernelMiscount}," + 
                                   f"{avgNonFluorABSDiff},{avgTotalKernelMiscount},{avgTotalABSDiff},{avgTransmissionDiff}," + 
                                   f"{avgTransmissionABSDiff},{inAvgFluorKernelMiscount},{inAvgFluorABSDiff}," + 
                                   f"{inAvgNonFluorKernelMiscount},{inAvgNonFluorABSDiff},{inAvgTotalKernelMiscount}," + 
                                   f"{inAvgTotalABSDiff},{inAvgTransmissionDiff},{inAvgTransmissionABSDiff}," +
                                   f"{f1FluorAvg},{f1NonFluorAvg}\n")

            
            torch.save(self.network.state_dict(), self.modelDir+"/EarVisionModel_"+str(e+1).zfill(3)+".pt")
            print("Saved " + "EarVisionModel_"+str(e+1).zfill(3)+".pt")

            print("\n~EPOCH "+ str(e+1) + " TRAINING COMPLETE~ \n")

        self.trainingLog.close()

