import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

style.use("ggplot")



def createGraph(modelDir):

    fileLines = open("SavedModels/"+modelDir+"/TrainingLog.tsv", "r").read().split('\n')

    epochs = []

    inLosses = []
    outLosses = []

    outFluorABSs = []
    outNonFluorABSs = []
    outTotalABSs = []


    stopAtEpoch = 30

    for line in fileLines[1:stopAtEpoch+2]: #start at 3rd line (ind 2) because 1st (ind 0) is column names, 2nd line (ind 1) are starting metrics (not ALWAYS helpful to graph those)  --NOT TRUE HERE, ADD STARTING METRICS
        if line != "":
            columns = line.split("\t")
            epoch, inLoss, outLoss= columns[0:3]
            epochs.append(int(epoch))

            inLosses.append(float(inLoss))
            outLosses.append(float(outLoss))

            outFluorABS = columns[4]
            outNonFluorABS = columns[6]
            outTotalABS = columns[8]

            outFluorABSs.append(float(outFluorABS))
            outNonFluorABSs.append(float(outNonFluorABS))
            outTotalABSs.append(float( outTotalABS ))


    lossPlot, l_ax = plt.subplots(1, 1, num='Loss')
    l_ax.plot(epochs, inLosses, label="In-Sample Loss")
    l_ax.plot(epochs, outLosses, label="Out-Sample Loss")
    l_ax.legend(loc=1)
    l_ax.set_xlabel("Epochs")
    l_ax.set_ylabel("Loss")
    l_ax.set_title("Loss over Training Epochs")
    #lossPlot.suptitle('Super title')
    lossPlot.savefig("loss_test.pdf")

    
    errorPlot, e_ax = plt.subplots(1,2, num="Error", sharey = True)

    e_ax[0].plot(epochs, outFluorABSs, '#afff00', label="Fluor")
    e_ax[0].plot(epochs, outNonFluorABSs, '#4c00e6', label="Non Fluor")
    e_ax[0].plot(epochs, outTotalABSs, '#000000', label="Total")
    e_ax[0].set_title("Out-Sample Avg Kernel Miscount (ABS) by Class")
    e_ax[0].yaxis.set_tick_params(labelbottom=True)
    e_ax[0].legend(loc="lower right")
    #d_ax[1].title.set_size(24)
    #d_ax[1].xaxis.label.set_size(20)
    #d_ax[1].yaxis.label.set_size(20)
    e_ax[0].set_xlabel("Epochs")

    errorPlot.savefig("error_plot.png")

    plt.show()

createGraph("02.21.23_03.39PM")
