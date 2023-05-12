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

        fab = torch.stack(fluorActBoxes)
        nfab = torch.stack(fluorActBoxes)
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
                        # break
                if not match:
                    falsePos += 1

        for box in boxIdx:
            if box not in foundBoxes:
                #print(f"ground truth box {box} is a false negative")
                falseNeg += 1

        boxIdx.clear()
        foundBoxes.clear()

        return truePos, falsePos, falseNeg
        

    def train(self):

        torch.set_grad_enabled(True) #enable the gradient
        epochs = self.hyperparameters["epochs"]
        # NOTE: above line is correct
        #epochs = 10
        '''
        goes from 0 to 1, the closer to 1 the better
        will need to isolate to compare -> print(MAP['map'])
        {
            'map': tensor(0.2822), 
            'map_50': tensor(0.5566), 
            'map_75': tensor(0.2484), 
            'map_small': tensor(0.0160), 
            'map_medium': tensor(0.2720), 
            'map_large': tensor(0.3679), 
            'mar_1': tensor(0.0038), 
            'mar_10': tensor(0.0374), 
            'mar_100': tensor(0.3427), 
            'mar_small': tensor(0.0126), 
            'mar_medium': tensor(0.3395), 
            'mar_large': tensor(0.4863), 
            'map_per_class': tensor(-1.), 
            'mar_100_per_class': tensor(-1.)
        }

        https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/mean_ap.py
        '''
        
        # NOTE: if we go back to using MAP, just do vMAP
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

            # __getitem__() here
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

            # MAP seems to be really time intensive
            
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

                    #do we need NMS???
                    finalPredictions = predictions

                    # labels = [None, "nonfluorescent",  "fluorescent"]
                    #            0           1                 2  

                    for k in range(1, len(finalPredictions)):           
                        '''
                        "k" is an image in validation set. Iterate through predictions in validation set images and 
                        calculate F1 scores for each image; append to list.
                        '''                        

                        #### TODO remove after testing
                        predictedFluorCnttest = p['labels'].tolist().count(2)
                        predictedNonFluorCnttest = p['labels'].tolist().count(1)
                        actualFluorCnttest = annotations[k]['labels'].tolist().count(2)
                        actualNonFluorCnttest = annotations[k]['labels'].tolist().count(1)
                        ######



                        if finalPredictions[k]["boxes"].size(dim = 1) != 0:
                        
                            fluorPredBoxes, fluorActBoxes, nonFluorPredBoxes, nonFluorActBoxes = self.splitBoxesByLabel(finalPredictions[k], annotations[k])

                            iousFluor, iousNonFluor = None, None
                            if fluorPredBoxes.size(dim = 0) != 0:
                                iousFluor = box_iou(fluorPredBoxes, fluorActBoxes)

                            if nonFluorPredBoxes.size(dim = 0) != 0:
                                iousNonFluor = box_iou(nonFluorPredBoxes, nonFluorActBoxes)   


                            truePosFluor, falsePosFluor, falseNegFluor = self.compPredsVsAnnotation(iousFluor, len(fluorActBoxes))
                            truePosNonFluor, falsePosNonFluor, falseNegNonFluor = self.compPredsVsAnnotation(iousNonFluor, len(nonFluorActBoxes))


                            f1FluorValues.append((2 * truePosFluor) / ((2 * truePosFluor) + falsePosFluor + falseNegFluor))


                            '''
                            TODO:

                            F1 for nonfluor is really low -->
                                    too many false positives or false negatives? 
                            '''


                            f1NonFluorValues.append((2 * truePosNonFluor) / ((2 * truePosNonFluor) + falsePosNonFluor + falseNegNonFluor))

                            # TODO: start by looking at these results for the most recent model; may help get an idea why the nonfluor vals are so wonky

                            print()
                            print(f"image {k} counts: {actualFluorCnttest} actual fluor kernels, {predictedFluorCnttest} predicted fluor kernels, {actualNonFluorCnttest} actual nonfluor kernels, {predictedNonFluorCnttest} predicted nonfluor kernels")
                            print(f"image {k} nonfluorescent kernels: {truePosNonFluor} true positive, {falsePosNonFluor} false positive, {falseNegNonFluor} false negative, F1 = {(2 * truePosNonFluor) / ((2 * truePosNonFluor) + falsePosNonFluor + falseNegFluor)}")
                            print(f"image {k} fluorescent kernels: {truePosFluor} true positive, {falsePosFluor} false positive, {falseNegFluor} false negative, F1 = {(2 * truePosFluor) / ((2 * truePosFluor) + falsePosFluor + falseNegFluor)}")
                            print()
                        
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

