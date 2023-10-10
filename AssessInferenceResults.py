'''
TODO:

figure out how to call r script rom here for each inference model cause i don'tfeel like hard coding every single one omg
'''


'''
EarVision 2.0:
Assess Inference Results

This script creates graphs from Inference Output data.
'''

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from Infer import Infer
from tkinter import Tk
from tkinter import *
import tkinter.filedialog as filedialog
import os

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress




actTrans, actFluor, actNonFluor = {}, {}, {}




def makeInteractive():
    homeDirec = os.getcwd()

    root = Tk()
    root.withdraw()
    inferenceDirectoryPath = filedialog.askdirectory(initialdir=homeDirec+"/Inference")

    print("Opening " + inferenceDirectoryPath)

    root.destroy()
    
def graphDriver(filename, num, filter):
    x, y = makeListsForGraph(filename, "PredictedTransmission", "ActualTransmission", filter)
    makeGraph(x, y, "PredictedTransmission", "ActualTransmission", f"X {num}", 0, 100, 0, 100)
    #makeGraph(filename, "AmbiguousKernels","TranmissionABSDiff", f"B {num}", 0, 35, 0, 35, filter)
    #makeGraph(filename, "PredictedTotal", "AmbiguousKernels", f"B {num}", 0, 600, 0, 35, filter)
    #x, y = makeListsForGraph(filename, "PredictedTotal", "TranmissionABSDiff", filter)
    #makeGraph(x, y, "PredictedTotal", "TranmissionABSDiff", f"B {num}", 0, 600, 0, 35)
    #makeGraph(filename, "PredictedFluor", "TranmissionABSDiff", f"B {num}", 0, 500, 0, 35)
    #makeGraph(filename, "PredictedNonFluor", "TranmissionABSDiff", f"B {num}", 0, 500, 0, 35)


def makeGraph(x, y, xLabel, yLabel, title, xmin, xmax, ymin, ymax):
    #x, y = makeListsForGraph(filename, xLabel, yLabel, filter)
    #print(y)
    '''
    correlationMatrix = np.corrcoef(x, y)
    corr = correlationMatrix[0,1]

    rsqd = corr ** 2
    print(f"corrcoef\t{rsqd}")

    rsqd2 = r2_score(x, y)
    print(f"r2_score\t{rsqd2}")

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print(f"r_value\t\t{r_value}")
    print(f"r_value^2\t{r_value**2}")

    model = LinearRegression()
    xArray = np.array(x)
    yArray = np.array(y)
    newXarr = xArray.reshape(-1, 1)
    #print(newXarr)

    model.fit(newXarr, yArray)
    print(f"model.score\t{model.score(newXarr, yArray)}")
    print(f"model.score adj\t{1 - (1-model.score(newXarr, yArray))*(len(yArray)-1)/(len(yArray)-newXarr.shape[1]-1)}")
    '''
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.scatter(x, y)
    plt.axis('square')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    #plt.axis('equal')
    plt.show()


def findProblemImages(filenames):
    ambigEars = {}
    for filename in filenames:
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            if row["TrainingSet"] != True:
                if row["EarName"] not in ambigEars:
                    ambigEars[row["EarName"]] = []
                ambigEars[row["EarName"]].append(row["AmbiguousKernels"])

    problemEars = {}
    for ear in ambigEars:
        for ambigs in ambigEars[ear]:
            if ambigs >= 8:
                if ear not in problemEars:
                    problemEars[ear] = []
                problemEars[ear] = ambigEars[ear]

    print(problemEars.keys())


def makeListsForGraph(filename, colNameX, colNameY, filter):
    xList, yList = [], []
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        if row["TrainingSet"] != True:
            if row["ActualTransmission"] > -1:
                if filter:
                    if row["PredictedTotal"] > 100 and row["AmbiguousKernels"] < 8:
                    #if row["PredictedTotal"] <= 100 or row["AmbiguousKernels"] >= 8:
                        #print(f"{row['EarName']}\t{row['PredictedTotal']}\t{row['AmbiguousKernels']}")
                        xList.append(row[colNameX])
                        yList.append(row[colNameY])
                else:
                    #print(f"{row['EarName']}\t{row['PredictedTotal']}\t{row['AmbiguousKernels']}")
                    xList.append(row[colNameX])
                    yList.append(row[colNameY])
                        
    return xList, yList

def numImgs(df):
    numImgsUnder200 = 0
    numImgsUnder150 = 0
    numImgsTotal = 0
    for index, row in df.iterrows():
        if row["TrainingSet"] != True:
            numImgsTotal += 1
            if row["PredictedTotal"] < 200:
                numImgsUnder200 += 1
                #numImgsOver150 += 1
            if row["PredictedTotal"] < 150:
                numImgsUnder150 += 1
    print(numImgsUnder200)
    print(numImgsUnder150)
    print(numImgsTotal)


def newInfData(df):
    predFluor, predNonFluor, predTrans = {}, {}, {}
    for index, row in df.iterrows():
        if row["TrainingSet"] != True:
            if row["ActualTransmission"] > -1:
                earName = row["EarName"]
                if earName not in predTrans:
                    predTrans[earName] = 0
                predTrans[earName] = row["PredictedTransmission"]
                if earName not in predFluor:
                    predFluor[earName] = 0
                predFluor[earName] = row["PredictedFluor"]
                if earName not in predNonFluor:
                    predNonFluor[earName] = 0
                predNonFluor[earName] = row["PredictedNonFluor"]

                if earName not in actTrans:
                    actTrans[earName] = row["ActualTransmission"]
                if actTrans[earName] != row["ActualTransmission"]:
                    print(f"issue with {earName} actual transmission")
                if earName not in actFluor:
                    actFluor[earName] = row["ActualFluor"]
                if actFluor[earName] != row["ActualFluor"]:
                    print(f"issue with {earName} actual fluorescent kernels")
                if earName not in actNonFluor:
                    actNonFluor[earName] = row["ActualNonFluor"]
                if actNonFluor[earName] != row["ActualNonFluor"]:
                    print(f"issue with {earName} actual nonfluor kernels")
    return predFluor, predNonFluor, predTrans
  

def makeWarmanCompGraphs(df, modelName, warmanList):
    # Create dicts where key is earname and val is number (numFluor, numNonFluor, trans)
    # This fuction also fills global structs for actual values
    predFluor, predNonFluor, predTrans = newInfData(df)
    yNewModPredFluor, yNewModPredNonFluor, yNewModPredTrans = [], [], []
    xActFluor, xActNonFluor, xActTrans = [], [], []
    # We only want to look at ears that were used in Warman's tests
    for ear in warmanList:
        # Make sure ear has actual vals associated with it (should always be true) and that it has model predicted values
        # in it (should also always be true)
        if ear in actFluor and ear in predFluor:
            # Make lists to use in graph, assoc. values added in the same order
            yNewModPredFluor.append(predFluor[ear])
            yNewModPredNonFluor.append(predNonFluor[ear])
            yNewModPredTrans.append(predTrans[ear])
            
            xActFluor.append(actFluor[ear])
            xActNonFluor.append(actNonFluor[ear])
            xActTrans.append(actTrans[ear])

    # Graph that stuff
    '''
    print(modelName)
    print(f"maxActFluor:{max(xActFluor)}\tmaxNewFluor:{max(yNewModPredFluor)}")
    print(f"maxActNonFluor:{max(xActNonFluor)}\tmaxNewNonFluor:{max(yNewModPredNonFluor)}")
    print(f"maxActTrans:{max(xActTrans)}\tmaxNewTrans:{max(yNewModPredTrans)}")
    print()
    '''
    makeGraph(xActFluor, yNewModPredFluor, "Actual Number of Fluorescent Kernels", "Predicted Number of Fluorescent Kernels", modelName + " Fluor", 0, 500, 0, 500)
    makeGraph(xActNonFluor, yNewModPredNonFluor, "Actual Number of NonFluorescent Kernels", "Predicted Number of NonFluorescent Kernels", modelName + " NonFluor", 0, 500, 0, 500)
    makeGraph(xActTrans, yNewModPredTrans, "Actual Transmission", "Predicted Transmission", modelName + " Transmission", 0, 100, 0, 100)


def checkFiles(warmanList):
    pngFiles = []
    xmlFiles = []
    for file in os.listdir("Inference\curatedImagesX"):
        if file.endswith(".png"):
            pngFiles.append(file.split(".")[0])
        if file.endswith(".xml"):
            xmlFiles.append(file.split(".")[0])
    for earname in warmanList:
        if earname not in pngFiles:
            print(f"{earname} does not have a .png image")
        if earname not in xmlFiles:
            print(f"{earname}")
    

def getImgList(filename):
    newList = []
    with open(filename, "r") as listfile:
        for line in listfile:
            earname = line.split(".")[0]
            newList.append(earname.strip())
    return newList

def getCSVfiles(dirname):
    infFilePaths = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.startswith("InferenceOutput") and file.endswith(".csv"):
                infFilePaths.append(os.path.join(root, file))
    return infFilePaths


def recreateWarmanGraphs(dfw, warmanList, dataset):
    warmanFluor, warmanNonFluor, warmanPredTrans = [], [], []
    warmanActFluor, warmanActNonFluor, warmanActTrans = [], [], []
    for index, row in dfw.iterrows():
        # if earname is in the testing set and has actual values associated with it (both should always be true)
        if row["image_name"] in warmanList and row["image_name"] in actFluor:
            warmanFluor.append(row["GFP_tf"])        # Grab fluor vals
            warmanNonFluor.append(row["wt_tf"])      # Grab nonfluor vals
            # Calculate Trans vals
            warmanPredTrans.append(row["GFP_tf"] / (row["wt_tf"] + row["GFP_tf"]) * 100)#

            # Make lists of associated actual values
            warmanActFluor.append(actFluor[row["image_name"]])
            warmanActNonFluor.append(actNonFluor[row["image_name"]])
            warmanActTrans.append(actTrans[row["image_name"]])   
    '''
    print(f"Warman {dataset}")
    print(f"maxWarmanFluor:{max(warmanFluor)}")
    print(f"maxWarmanNonFluor:{max(warmanNonFluor)}")
    print(f"maxWarmanTrans:{max(warmanPredTrans)}")
    print()
    '''

    makeGraph(warmanActFluor, warmanFluor, "Actual Number of Fluorescent Kernels", "Predicted Number of Fluorescent Kernels", f"Warman {dataset} Fluor", 0, 400, 0, 400)
    makeGraph(warmanActNonFluor, warmanNonFluor, "Actual Number of NonFluorescent Kernels", "Predicted Number of NonFluorescent Kernels", f"Warman {dataset} NonFluor", 0, 400, 0, 400)
    makeGraph(warmanActTrans, warmanPredTrans, "Actual Transmission", "Predicted Transmission", f"Warman {dataset} Transmission", 0, 100, 0, 100)

    # TODO: actually make and save graphs!!!





def main():

    # Get lists of all images that were in each testing set, from https://datacommons.cyverse.org/browse/iplant/home/shared/EarVision_maize_kernel_image_data/testing_images
    warmanX = getImgList("Inference/2018X_earLIst.txt")
    warmanY = getImgList("Inference/2019Y_earList.txt")

    # Make lists of PATHS to InferenceOutput csv files
    yInfs = getCSVfiles("Inference/testingSetWarmanPaperY")
    xInfs = getCSVfiles("Inference/testingSetWarmanPaperX")

    # Extract data from inference csv files and graph results
    # These graphs will have the new inference data for each inference, but will only contain image data that was used 
    # in Fig 5 of Warman's paper (see above for datacommons link) 
    for yFile in yInfs:
        df = pd.read_csv(yFile)
        modelName = yFile.split("\\")[-2].split("_")[-1]
        makeWarmanCompGraphs(df, modelName + " X", warmanY)

    for xFile in xInfs:
        df = pd.read_csv(xFile)
        modelName = xFile.split("\\")[-2].split("_")[-1]
        makeWarmanCompGraphs(df, modelName + " Y", warmanX)

    # This data was used in Figure 5 of the Warman paper, found at https://github.com/fowler-lab-osu/maize_ear_scanner_and_computer_vision_statistics/blob/master/data/test_set_two_models_predictions_2018_summary.tsv
    # and https://github.com/fowler-lab-osu/maize_ear_scanner_and_computer_vision_statistics/blob/master/data/test_set_two_models_predictions_2019_summary.tsv
    dfwx = pd.read_csv("Inference/testingSetWarmanPaperX/test_set_two_models_predictions_2018_summary.tsv", sep="\t")
    dfwy = pd.read_csv("Inference/testingSetWarmanPaperY/test_set_two_models_predictions_2019_summary.tsv", sep="\t")

    # Recreate Warman's Fig. 5 results for as close to a one-to-one comparison as possible
    #print("X")
    recreateWarmanGraphs(dfwx, warmanX, "2018 X")
    print()
    #print("Y")
    recreateWarmanGraphs(dfwy, warmanY, "2019 Y")

    # TODO: what other stats can I glean from his data? transDiff, transABSDIff, predActRatio, etc?

    # TODO: make this user friendly for future data analysis? for when I'm bored or really sick of working on higher priority stuff
    # This works!! maybe make it so someone could select the inference data they want graphs on?
    # makeInteractive()

    #for inf in yInfs:
    #    #print(inf)
    #    print("calling r script")
    #    subprocess.call(f"rscript recreateWarmanData.r {inf}", shell=True)
    #    # TODO: why does this not work???

main()