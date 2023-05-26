import os
import re
import shutil


'''
all crosses are female by male



Male

    B2Ex54-1_m1
    B2x52-7
    B2x286-4_m1


Female
    B52-1x2
    B104-6x2
    B210-4x2
    BUNK-10x2


'''



def makeNewDirs(trainingImageDir, newDirNames):
    os.makedirs(trainingImageDir + "SplitBySex", exist_ok=True)
    for name in newDirNames:
        os.makedirs(trainingImageDir + "SplitBySex/" + name, exist_ok=True)



def separateImages(trainingImageDir):

    newDirNames = ["AnnosFemale/", "AnnosMale/", "AnnosUnsure/", "ImagesFemale/", "ImagesMale/", "ImagesUnsure/", "WhatAreThese/"]

    makeNewDirs(trainingImageDir, newDirNames)

    newDirNames = [name + "EarDataset/All/SplitBySex/" for name in newDirNames]
    femaleAnno, maleAnno, unsureAnno = [], [], []
    femaleImgs, maleImgs, unsureImgs = [], [], []
    female, male, unsure = [], [], []
    for root, dirs, files in os.walk(trainingImageDir):
        for name in files:
            parents = name.split(".")[0]
            try:
                mother, father = name.split("x")
                if mother.find("-") != -1 and father.find("-") != -1:
                    unsure.append(name)
                elif mother.find("-") != -1 and father.find("-") == -1:
                    female.append(name)
                elif mother.find("-") == -1 and father.find("-") != -1:
                    male.append(name)
                else:
                    unsure.append(name)
            except:
                unsure.append(name)


#dest = shutil.copy2(source, destination)


    for file in female:
        if file.endswith((".xml")):
            x = shutil.copy2("EarDataset/All/Images/" + file, newDirNames[0] + file)
        elif file.endswith((".png")):
            pass
        else:
            pass

    if father.endswith((".xml")):
        pass
    if father.endwith((".png")):
        pass
    print(f"Female: {female}")
    print(f"Male: {male}")
    print(f"Unsure: {unsure}")

if __name__ == "__main__":
    separateImages("EarDataset/All/")