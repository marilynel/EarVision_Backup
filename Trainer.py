import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import datetime
import os



class Trainer():
    def __init__(self, network, trainDataLoader, validationDataLoader, device):
        self.device = device
        self.network = network
        ''' have seen this in some places:    
         params = [p for p in network.parameters() if p.requires_grad], when passing params to optimizer. 
         only calculates trainable parameters '''
        #consider different optimizers here as well
        self.optimizer = optim.Adam(network.parameters(), lr = 0.0004)
        '''note: i have not been using a Learning Rate Scheduler, since I
        have been using Adam. but it might be work trying, either with Adam or
        with a differnt optimizer (like SGD) ''' 
        #look into trying torch.optim.lr_scheduler.StepLR
        self.trainingLoader = trainDataLoader
        self.validationLoader = validationDataLoader
        self.classNum = 4 #three fruit classes + background

        startTime = datetime.datetime.now()
        self.modelDir = startTime.strftime("%m.%d.%y_%I.%M%p")
        
        if not os.path.isdir("SavedModels"):
            os.mkdir("SavedModels")
        os.mkdir("SavedModels/"+self.modelDir)



    def train(self):

        torch.set_grad_enabled(True) #enable the gradient

        epochs = 5
        for e in range(epochs):
            self.network.train() #put network in train mode

            print("------------------")
            print("COMMENCING EPOCH: ", str(e+1)+"/"+str(epochs) )
            print("------------------")

            for batch in tqdm(self.trainingLoader):
                images, annotations = batch
                #weirdly, the F-RCNN model requires inputs in form of list
                #so we gotta turn the batch tensors into lists, and also send each separate item to GPU
                images = list(image.to(self.device) for image in images)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]



                self.network.zero_grad()
                # consider :  with torch.cuda.amp.autocast(enabled=scaler is not None):?
                lossDict = self.network(images, annotations)
                losses = sum(loss for loss in lossDict.values())
                #print(losses)

                losses.backward()
                self.optimizer.step()

