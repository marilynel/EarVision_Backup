import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import datetime
import os
from Utils import *


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


        self.trainingLog = open(self.modelDir + "/TrainingLog.txt", "w+")
        self.trainingLog.write("Epoch"+"\t"+"InSampleLoss"+"\t" + "OutSampleLoss" + "\t")
        #self.trainingLog.write("InSampleError" + "\t" + "OutSampleError" + "\n")
        self.trainingLog.write("OutFluorKernelDiff" + "\t" + "OutFluorError" + "\t" + 
        "OutNonFluorKernelDiff"+ "\t" "OutNonFluorError"+ "\t" +
        "OutTotalKernelDiff" + "\t" "OutTotalError" + "\t" + "OutTransmissionDiff" + "\t" + "OutTransmissionError" "\n")


    def forwardPass(self, images, annotations, train=False, trackMetrics = True):
        if train:
            #zero the gradient
            self.network.zero_grad()
        
        modelOutput = self.network(images, annotations)
        return modelOutput


    def train(self):

        torch.set_grad_enabled(True) #enable the gradient

        epochs = self.hyperparameters["epochs"]
        for e in range(epochs):
            print("------------------")
            print("COMMENCING EPOCH: ", str(e+1)+"/"+str(epochs) )
            print("------------------")

            totalFluorErrorTr = 0
            totalNonFluorErrorTr = 0
            totalTotalErrorTr = 0
            trainingLossTotal = 0

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
                    finalPredictions = self.network(images)
                    for i, p in enumerate(finalPredictions):
            
                        predictedFluorCnt = p['labels'].tolist().count(2)
                        predictedNonFluorCnt = p['labels'].tolist().count(1)
                        actualFluorCnt = annotations[i]['labels'].tolist().count(2)
                        actualNonFluorCnt = annotations[i]['labels'].tolist().count(1)

                        countMetrics = calculateCountMetrics([ predictedFluorCnt, predictedNonFluorCnt], [actualFluorCnt, actualNonFluorCnt])
                        
                        fluorKernelDiff = countMetrics[0]
                        fluorPercentageDiff = countMetrics[1]

                        nonFluorKernelDiff= countMetrics[2]
                        nonFluorPercentageDiff = countMetrics[3]

                        totalKernelDiff = countMetrics[4]
                        totalPercentageDiff = countMetrics[5]

                        transmissionDiff = countMetrics[6]
                        transmissionPercentageDiff = countMetrics[7]

                        totalFluorErrorTr += fluorPercentageDiff 
                        totalNonFluorErrorTr += nonFluorPercentageDiff
                        totalTotalErrorTr += totalPercentageDiff

            avgFluorErrorTr = totalFluorErrorTr /  (len(self.trainingLoader) * self.trainingLoader.batch_size)
            avgNonFluorErrorTr = totalNonFluorErrorTr / (len(self.trainingLoader) * self.trainingLoader.batch_size)
            avgTotalErrorTr = totalTotalErrorTr / (len(self.trainingLoader) * self.trainingLoader.batch_size)

            avgTrainingLoss = (trainingLossTotal / len(self.trainingLoader))
            print("Training loss:", avgTrainingLoss )
            print("Training Errors: ")
            print("-Avg Fluor Error: ", avgFluorErrorTr)
            print("-Avg NonFluor Error:", avgNonFluorErrorTr)
            print("-Avg Total Error: ", avgTotalErrorTr)


            print("VALIDATING")
            #because of quirky reasons, the network needs to be in .train() mode to get the validation loss... silly
            #but it should be kosher since there are no dropout layers and the BatchNorms are frozen
            
            #self.network.eval()
            with torch.no_grad():
                
                totalFluorKernelDiff = 0
                totalFluorError = 0

                totalNonFluorKernelDiff = 0
                totalNonFluorError = 0
                totalTotalKernelDiff = 0
                totalTotalError = 0

                totalTransmissionDiff = 0
                totalTransmissionError = 0

                validationLossTotal = 0

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

                    for i, p in enumerate(finalPredictions):
                        
                        predictedFluorCnt = p['labels'].tolist().count(2)
                        predictedNonFluorCnt = p['labels'].tolist().count(1)

                        actualFluorCnt = annotations[i]['labels'].tolist().count(2)
                        actualNonFluorCnt = annotations[i]['labels'].tolist().count(1)

                        countMetrics = calculateCountMetrics([ predictedFluorCnt, predictedNonFluorCnt], [actualFluorCnt, actualNonFluorCnt])
                        
                        fluorKernelDiff = countMetrics[0]
                        fluorPercentageDiff = countMetrics[1]

                        nonFluorKernelDiff= countMetrics[2]
                        nonFluorPercentageDiff = countMetrics[3]

                        totalKernelDiff = countMetrics[4]
                        totalPercentageDiff = countMetrics[5]

                        transmissionDiff = countMetrics[6]
                        transmissionPercentageDiff = countMetrics[7]


                        #continue summing the metrics:
                        totalFluorKernelDiff += fluorKernelDiff
                        totalFluorError += fluorPercentageDiff 

                        totalNonFluorKernelDiff += nonFluorKernelDiff
                        totalNonFluorError += nonFluorPercentageDiff

                        totalTotalKernelDiff += totalKernelDiff
                        totalTotalError += totalPercentageDiff

                        totalTransmissionDiff += transmissionDiff
                        totalTransmissionError += transmissionPercentageDiff

            earImageCount =( len(self.validationLoader) * self.validationLoader.batch_size)

            avgFluorKernelMiscount = totalFluorKernelDiff / earImageCount
            avgFluorError = totalFluorError /  earImageCount

            avgNonFluorKernelMiscount = totalNonFluorKernelDiff / earImageCount
            avgNonFluorError = totalNonFluorError / earImageCount

            avgTotalKernelMiscount = totalTotalKernelDiff / earImageCount
            avgTotalError = totalTotalError / earImageCount

            avgTransmissionDiff = totalTransmissionDiff / earImageCount
            avgTransmissionError = totalTransmissionError / earImageCount

            avgValidationLoss = validationLossTotal /  (len(self.validationLoader) )  

            print("Validation Loss: " , avgValidationLoss)
            print("Validation Errors: ")
            print("-Avg Fluor Kernel Miscount: ", avgFluorKernelMiscount)
            print("-Avg Fluor Error: ", avgFluorError)

            print("-Avg NonFluor Kernel Miscount: ", avgNonFluorKernelMiscount)
            print("-Avg NonFluor Error:", avgNonFluorError)

            print("-Avg Total Kernel Miscount: ",  avgTotalKernelMiscount)
            print("-Avg Total Error: ", avgTotalError)

            print("-Avg Transmission Diff: ", avgTransmissionDiff)
            print("-Avg Transmission Error: ", avgTransmissionError)

            print('----')
                
            self.trainingLog.writelines([str(e+1)+"\t" , str(avgTrainingLoss.item()) +"\t", str(avgValidationLoss.item())+"\t"])
            '''
            for i in range(5):
                self.trainingLog.writelines([str(inSampleMetrics[i])+"\t", str(outSampleMetrics[i])+"\t"])
            '''
            self.trainingLog.writelines([str(avgFluorKernelMiscount)+"\t", str(avgFluorError)+"\t", 
            str(avgNonFluorKernelMiscount)+"\t", str(avgNonFluorError)+"\t",
            str(avgTotalKernelMiscount) +"\t", str(avgTotalError) +"\t", str(avgTransmissionDiff) +"\t", str(avgTransmissionError)])
            self.trainingLog.writelines(["\n"])


            torch.save(self.network.state_dict(), self.modelDir+"/EarVisionModel_"+str(e+1).zfill(3)+".pt")
            print("Saved " + "EarVisionModel_"+str(e+1).zfill(3)+".pt")

            print("\n~EPOCH "+ str(e+1) + " TRAINING COMPLETE~ \n")

        self.trainingLog.close()