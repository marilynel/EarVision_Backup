import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import datetime
import os
from Utils import *

from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
        self.trainingLog.write("Epoch"+","+"InSampleLoss"+"," + "OutSampleLoss" + ",")
        #self.trainingLog.write("InSampleError" + "\t" + "OutSampleError" + "\n")
        '''
        self.trainingLog.write("OutFluorKernelDiff" + "\t" + "OutFluorError" + "," + 
        "OutNonFluorKernelDiff"+ "\t" "OutNonFluorError"+ "\t" +
        "OutTotalKernelDiff" + "\t" "OutTotalError" + "\t" + "OutTransmissionDiff" + "\t" + "OutTransmissionError" "\n")
        '''
        # TODO: reorder this so ins and outs are together?
        self.trainingLog.write("OutFluorKernelDiff" + "," + "OutFluorABSDiff" + "," + 
        "OutNonFluorKernelDiff"+ "," "OutNonFluorKernelABSDiff"+ "," +
        "OutTotalKernelDiff" + "," "OutTotalKernelABSDiff" + "," + "OutTransmissionDiff" + "," + "OutTransmissionABSDiff" + "," +
        "InFluorKernelDiff" + "," + "InFluorABSDiff" + "," + 
        "InNonFluorKernelDiff"+ "," "InNonFluorKernelABSDiff"+ "," +
        "InTotalKernelDiff" + "," "InTotalKernelABSDiff" + "," + "InTransmissionDiff" + "," + "InTransmissionABSDiff" + "\n")


    def forwardPass(self, images, annotations, train=False, trackMetrics = True):
        if train:
            #zero the gradient
            self.network.zero_grad()
        
        modelOutput = self.network(images, annotations)
        return modelOutput

    ### NEW ###
    def print_metrics(self, avgFluorKernelMiscount, avgFluorABSDiff, avgNonFluorKernelMiscount, avgNonFluorABSDiff, avgTotalKernelMiscount, avgTotalABSDiff, avgTransmissionDiff, avgTransmissionABSDiff):
        print("-Avg Fluor Kernel Miscount: ", avgFluorKernelMiscount)
        #print("-Avg Fluor Error: ", avgFluorError)
        print("-Avg Fluor Kernel Miscount (Absolute Value): ", avgFluorABSDiff)

        print("-Avg NonFluor Kernel Miscount: ", avgNonFluorKernelMiscount)
        #print("-Avg NonFluor Error:", avgNonFluorError)
        print("-Avg NonFluor Kernal Miscount (Absolute Value):", avgNonFluorABSDiff)

        print("-Avg Total Kernel Miscount: ",  avgTotalKernelMiscount)
        #print("-Avg Total Error: ", avgTotalError)
        print("-Avg Total Kernal Miscount (Absolute Value):", avgTotalABSDiff)


        print("-Avg Transmission Diff: ", avgTransmissionDiff)
        #print("-Avg Transmission Error: ", avgTransmissionError)
        print("-Avg Transmission Diff (Absolute Value):", avgTransmissionABSDiff)

        print('----')
    ### END NEW ###

    def train(self):

        torch.set_grad_enabled(True) #enable the gradient
        epochs = self.hyperparameters["epochs"]

        trainingMAP = MeanAveragePrecision()
        #validationMAP = MeanAveragePrecision(class_metrics = True)
        validationMAP = MeanAveragePrecision()

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
                #print("Loss sum: ", lossSum)

                trainingLossTotal += lossSum
                #trainingLossTotal += lossMean

                lossSum.backward()
                #lossMean.backward()
              

                self.optimizer.step()

                self.network.eval()
                with torch.no_grad():
                    #the reason we run this through a second time just to get the outputs is due to a PyTorch F-CNN implementation quirk -- needs to be in eval() mode to get actual outputs.
                    finalPredictions = self.network(images) 

                    #print("Predictions: ", finalPredictions[0]['boxes'])
                    #print("Targets: ", annotations[0]['boxes'])
                    
                    #trainingMAP.update(finalPredictions, annotations)

                    for i, p in enumerate(finalPredictions):
                        predictedFluorCnt = p['labels'].tolist().count(2)
                        predictedNonFluorCnt = p['labels'].tolist().count(1)
                        actualFluorCnt = annotations[i]['labels'].tolist().count(2)
                        actualNonFluorCnt = annotations[i]['labels'].tolist().count(1)

                        countMetrics = calculateCountMetrics([ predictedFluorCnt, predictedNonFluorCnt], [actualFluorCnt, actualNonFluorCnt])
                        # return val of calculateCountMetrics
                        # metricList = [fluorKernelDiff, fluorKernelABSDiff, nonFluorKernelDiff, nonFluorKernelABSDiff, totalKernelDiff, totalKernelABSDiff, transmissionDiff, transmissionABSDiff]

                        fluorKernelDiff = countMetrics[0]
                        fluorKernelABSDiff = countMetrics[1]

                        nonFluorKernelDiff= countMetrics[2]
                        nonFluorKernelABSDiff = countMetrics[3]

                        totalKernelDiff = countMetrics[4]
                        totalKernelABSDiff = countMetrics[5]

                        transmissionDiff = countMetrics[6]
                        transmissionABSDiff = countMetrics[7]

                        inTotalFluorKernelDiff += fluorKernelDiff
                        inTotalFluorKernelABSDiff += fluorKernelABSDiff 

                        inTotalNonFluorKernelDiff += nonFluorKernelDiff
                        inTotalNonFluorKernelABSDiff += nonFluorKernelABSDiff

                        inTotalTotalKernelDiff += totalKernelDiff
                        inTotalTotalABSDiff += totalKernelABSDiff

                        inTotalTransmissionDiff += transmissionDiff
                        inTotalTransmissionABSDiff += transmissionABSDiff


            #avgFluorErrorTr = totalFluorErrorTr /  (len(self.trainingLoader) * self.trainingLoader.batch_size)
            #avgNonFluorErrorTr = totalNonFluorErrorTr / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            #avgTotalErrorTr = totalTotalErrorTr / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            ### NEW ###          
            inAvgFluorKernelMiscount = inTotalFluorKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgFluorABSDiff = inTotalFluorKernelABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgNonFluorKernelMiscount = inTotalNonFluorKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgNonFluorABSDiff = inTotalNonFluorKernelABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgTotalKernelMiscount = inTotalTotalKernelDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgTotalABSDiff = inTotalTotalABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            inAvgTransmissionDiff = inTotalTransmissionDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            inAvgTransmissionABSDiff = inTotalTransmissionABSDiff / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            ### END NEW ###


            avgTrainingLoss = (trainingLossTotal / len(self.trainingLoader))

            #print("computing tMAP....")
            #tMAP = trainingMAP.compute()
            #trainingMAP.reset()
            print("Training loss:", avgTrainingLoss )
          
            #print("Traning mAP: ", tMAP['map'])
            print("Training Errors: ")

            ### NEW ###            
            self.print_metrics(inAvgFluorKernelMiscount, inAvgFluorABSDiff, inAvgNonFluorKernelMiscount, inAvgNonFluorABSDiff, inAvgTotalKernelMiscount, inAvgTotalABSDiff, inAvgTransmissionDiff, inAvgTransmissionABSDiff)


            #print("-Avg Fluor Error: ", avgFluorErrorTr)
            #print("-Avg NonFluor Error:", avgNonFluorErrorTr)
            #print("-Avg Total Error: ", avgTotalErrorTr)


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


                ### NEW ###
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
                    
                    #print("p", finalPredictions[0]['boxes'], "a:", annotations[0]['boxes'])

                    #validationMAP.update(finalPredictions, annotations)


                    for i, p in enumerate(finalPredictions):
                        # repeated from above, can this be combined with previous codE?
                        predictedFluorCnt = p['labels'].tolist().count(2)
                        predictedNonFluorCnt = p['labels'].tolist().count(1)

                        actualFluorCnt = annotations[i]['labels'].tolist().count(2)
                        actualNonFluorCnt = annotations[i]['labels'].tolist().count(1)

                        countMetrics = calculateCountMetrics([ predictedFluorCnt, predictedNonFluorCnt], [actualFluorCnt, actualNonFluorCnt])
                        
                        fluorKernelDiff = countMetrics[0]
                        fluorKernelABSDiff = countMetrics[1]

                        nonFluorKernelDiff= countMetrics[2]
                        nonFluorKernelABSDiff = countMetrics[3]

                        totalKernelDiff = countMetrics[4]
                        totalKernelABSDiff = countMetrics[5]

                        transmissionDiff = countMetrics[6]
                        transmissionABSDiff = countMetrics[7]

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

            #valMAP = validationMAP.compute()
            #validationMAP.reset()
            #print("Validation mAP: " , valMAP)

            print("Validation Loss: " , avgValidationLoss)
            print("Validation Errors: ")
            
            ### NEW ###
            self.print_metrics(avgFluorKernelMiscount, avgFluorABSDiff, avgNonFluorKernelMiscount, avgNonFluorABSDiff, avgTotalKernelMiscount, avgTotalABSDiff, avgTransmissionDiff, avgTransmissionABSDiff)

                
            self.trainingLog.writelines([str(e+1)+"," , str(avgTrainingLoss.item()) +",", str(avgValidationLoss.item())+","])
            '''
            for i in range(5):
                self.trainingLog.writelines([str(inSampleMetrics[i])+"\t", str(outSampleMetrics[i])+"\t"])
            '''
            ### NEW ###
            self.trainingLog.writelines([str(avgFluorKernelMiscount)+",", str(avgFluorABSDiff)+",", 
            str(avgNonFluorKernelMiscount)+",", str(avgNonFluorABSDiff)+",",
            str(avgTotalKernelMiscount) +",", str(avgTotalABSDiff) +",", str(avgTransmissionDiff) +",", str(avgTransmissionABSDiff) + ",",
            str(inAvgFluorKernelMiscount)+",", str(inAvgFluorABSDiff)+",", 
            str(inAvgNonFluorKernelMiscount)+",", str(inAvgNonFluorABSDiff)+",",
            str(inAvgTotalKernelMiscount) +",", str(inAvgTotalABSDiff) +",", str(inAvgTransmissionDiff) +",", str(inAvgTransmissionABSDiff)])

            self.trainingLog.writelines(["\n"])
            ### END NEW ###

            torch.save(self.network.state_dict(), self.modelDir+"/EarVisionModel_"+str(e+1).zfill(3)+".pt")
            print("Saved " + "EarVisionModel_"+str(e+1).zfill(3)+".pt")

            print("\n~EPOCH "+ str(e+1) + " TRAINING COMPLETE~ \n")

        self.trainingLog.close()
    
    