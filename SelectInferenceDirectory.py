from Infer import Infer
from tkinter import Tk
from tkinter import *
import tkinter.filedialog as filedialog
import os


homeDirec = os.getcwd()

root = Tk()
root.withdraw()
inferenceDirectoryPath = filedialog.askdirectory(initialdir=homeDirec+"/Inference")

print("Opening " + inferenceDirectoryPath)

root.destroy()
Infer("03.06.23_12.55PM", "022", inferenceDirectoryPath)