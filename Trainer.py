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
from torchmetrics import F1Score



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
                               "InTransmissionABSDiff,tMAP,valMAP\n")


    def forwardPass(self, images, annotations, train=False, trackMetrics = True):
        if train:
            #zero the gradient
            self.network.zero_grad()

        modelOutput = self.network(images, annotations)

        return modelOutput


    def print_metrics(self, avgFluorKernelMiscount, avgFluorABSDiff, avgNonFluorKernelMiscount, avgNonFluorABSDiff, avgTotalKernelMiscount, avgTotalABSDiff, avgTransmissionDiff, avgTransmissionABSDiff):
        print("-Avg Fluor Kernel Miscount: ", avgFluorKernelMiscount)
        print("-Avg Fluor Kernel Miscount (Absolute Value): ", avgFluorABSDiff)

        print("-Avg NonFluor Kernel Miscount: ", avgNonFluorKernelMiscount)
        print("-Avg NonFluor Kernal Miscount (Absolute Value):", avgNonFluorABSDiff)

        print("-Avg Total Kernel Miscount: ",  avgTotalKernelMiscount)
        print("-Avg Total Kernal Miscount (Absolute Value):", avgTotalABSDiff)

        print("-Avg Transmission Diff: ", avgTransmissionDiff)
        print("-Avg Transmission Diff (Absolute Value):", avgTransmissionABSDiff)

        print('----')


    def train(self):

        torch.set_grad_enabled(True) #enable the gradient
        #epochs = self.hyperparameters["epochs"]
        # NOTE: above line is correct
        epochs = 10

        # goes from 0 to 1, the closer to 1 the better
        # will need to isolate to compare -> print(MAP['map'])
        # {
        #   'map': tensor(0.2822), 
        #   'map_50': tensor(0.5566), 
        #   'map_75': tensor(0.2484), 
        #   'map_small': tensor(0.0160), 
        #   'map_medium': tensor(0.2720), 
        #   'map_large': tensor(0.3679), 
        #   'mar_1': tensor(0.0038), 
        #   'mar_10': tensor(0.0374), 
        #   'mar_100': tensor(0.3427), 
        #   'mar_small': tensor(0.0126), 
        #   'mar_medium': tensor(0.3395), 
        #   'mar_large': tensor(0.4863), 
        #   'map_per_class': tensor(-1.), 
        #   'mar_100_per_class': tensor(-1.)
        # }

        # https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/mean_ap.py
        # NOTE: if we go back to using MAP, just do vMAP
        #trainingMAP = MeanAveragePrecision(iou_thresholds=[0.5,0.75],class_metrics=False)
        #validationMAP = MeanAveragePrecision(class_metrics = True)
        #validationMAP = MeanAveragePrecision(iou_thresholds=[0.5,0.75],class_metrics=False)

        for e in range(epochs):

            print("------------------")
            print("COMMENCING EPOCH: ", str(e+1)+"/"+str(epochs) )
            print("------------------")

            trainingLossTotal = 0
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


                    
                    # TODO: take this out later?
                    #print("Predictions: ", finalPredictions[0]['boxes'])
                    #print("Targets: ", annotations[0]['boxes'])
                    
                    for i, p in enumerate(finalPredictions):
                        # TODO: for new ambiguous labels, could grab here
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


                        # TODO: F1 Score
                        #target = torch.tensor()
                        #preds = torch.tensor()
                        # is this line usefule? f1_score(preds, target, task="multiclass", num_classes=3)
                        #f1score = torchmetrics.F1Score(task="multiclass", num_classes=2) #, average=None)
                        #
                        #f1score.update(preds, target)

                        


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
                inAvgTotalKernelMiscount, inAvgTotalABSDiff, inAvgTransmissionDiff, inAvgTransmissionABSDiff
            )

            print("VALIDATING")
            #because of quirky reasons, the network needs to be in .train() mode to get the validation loss... silly
            #but it should be kosher since there are no dropout layers and the BatchNorms are frozen
            
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
                    # do we need NMS???
                    finalPredictions = predictions

                    ious = box_iou(finalPredictions[0]['boxes'], annotations[0]['boxes'])

                    print(ious)

                    listActBoxes = [i for i in range(0, len(annotations[0]['boxes']))]
                    foundActBoxes = []

                    tp, fp, fn = 0, 0, 0

                    for i in range(0, len(ious)):
                        match = False
                        # print(f"i: {i}")
                        for j in range(0, len(ious[i])):
                            # print(f"j: {j}")
                            if ious[i][j] >= 0.7:
                                print(f"prediction {i} is a true positive for {listActBoxes[j]}")
                                match = True
                                foundActBoxes.append(listActBoxes[j])
                                tp += 1
                                # break
                        if not match:
                            print(f"prediction {i} is a false positive")
                            fp += 1

                    for box in listActBoxes:
                        if box not in foundActBoxes:
                            print(f"ground truth box {box} is a false negative")
                            fn += 1

                    print(f"True positives: {tp}")
                    print(f"False positives: {fp}")
                    print(f"False negatives: {fn}")

                    print()

                    f1score = (2 * tp) / ((2 * tp) + fp + fn)

                    print(f"F1 score: {f1score}")

                    # TODO: take back out later?
                    #print("p", finalPredictions[0]['boxes'], "a:", annotations[0]['boxes'])

                    #validationMAP.update(finalPredictions, annotations)


                    for i, p in enumerate(finalPredictions):
                        # TODO: F1NOTE collect predicted and anno boxes here? to calculate f1 score.  where can I get IOU? https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
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


            print("Validation Loss: " , avgValidationLoss)
            print("Validation Errors: ")
            
            self.print_metrics(avgFluorKernelMiscount, avgFluorABSDiff, avgNonFluorKernelMiscount, avgNonFluorABSDiff, avgTotalKernelMiscount, avgTotalABSDiff, avgTransmissionDiff, avgTransmissionABSDiff)
                
            self.trainingLog.writelines([str(e+1)+"," , str(avgTrainingLoss.item()) +",", str(avgValidationLoss.item())+","])
            

            self.trainingLog.write(f"{avgFluorKernelMiscount},{avgFluorABSDiff},{avgNonFluorKernelMiscount}," + 
                                   f"{avgNonFluorABSDiff},{avgTotalKernelMiscount},{avgTotalABSDiff},{avgTransmissionDiff}," + 
                                   f"{avgTransmissionABSDiff},{inAvgFluorKernelMiscount},{inAvgFluorABSDiff}," + 
                                   f"{inAvgNonFluorKernelMiscount},{inAvgNonFluorABSDiff},{inAvgTotalKernelMiscount}," + 
                                   f"{inAvgTotalABSDiff},{inAvgTransmissionDiff},{inAvgTransmissionABSDiff},NA,NA\n")

            
            torch.save(self.network.state_dict(), self.modelDir+"/EarVisionModel_"+str(e+1).zfill(3)+".pt")
            print("Saved " + "EarVisionModel_"+str(e+1).zfill(3)+".pt")

            print("\n~EPOCH "+ str(e+1) + " TRAINING COMPLETE~ \n")

        self.trainingLog.close()

