'''
EarVision 2.0:
TestHyperparameterSearchResults

This script performs inferences on specified datasets using multiple models. It can be used when the user wants to test 
multiple models against each other with the same or multiple datasets. "NOTE" will be in the comments above places where 
the code can be easily customized to the current needs of the user.
'''

from Infer import Infer
import os
import logging

homeDirec = os.getcwd()

# NOTE: Add or remove directories to run inference on
datasetsToTest = [
    #os.path.join(homeDirec,"Inference\curatedImagesA"),
    #os.path.join(homeDirec,"Inference\curatedImagesY")
    #os.path.join(homeDirec,"Inference\curatedImagesY"),
    #os.path.join(homeDirec,"Inference\curatedImagesZ"),
    #os.path.join(homeDirec,"Inference/xEars_2018")
    os.path.join(homeDirec, "Inference/testingSetWarmanPaperY")#,
    #os.path.join(homeDirec, "Inference/testingSetWarmanPaperX")
]

# NOTE: Add or remove models to test. Each sublist contains first the model identifier, then the epoch. 
modelsToTest = [
    ["08.18.22_07.13PM", "021"],
    ["03.06.23_12.55PM", "022"],        # 0306 22
    ["06.29.23_05.22PM", "025"],         # Atari 25
    ["06.29.23_09.50PM", "028"],          # Sunset 28
    ["06.22.23_04.15PM", "023"],        # Elmo 23
    ["06.23.23_01.33PM", "030"],        # LakeJabbul 30
    ["06.23.23_09.05AM", "029"],        # Bujanovac 29
    ["06.22.23_05.21PM", "030"]         # Fernando 30
]

# NOTE: Customize outfolder and log names here. Output will appear in this folder, within Inference/{dataset}/.
outFolder = ""
logName = "yEarsWithSomeModels.log"


for dataset in datasetsToTest:
    os.makedirs(f"{dataset}", exist_ok = True)
    logging.basicConfig(filename=f"{dataset}/{logName}", level=logging.INFO)

    for model in modelsToTest:  
        try:
            # NOTE: be sure to update the directories where the model may be found.
            modelID = f"{model[0]}"
            logging.info(f"Testing model {modelID} at epoch {model[1]}")
            Infer(modelID, model[1], dataset)
        except:
            logging.exception("")