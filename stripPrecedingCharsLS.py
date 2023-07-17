import os
import sys

def stripPrecedingCharsLS(directory):
    '''strips the preceding characters added to images and annotation files when exporting from labelstudio. Run on a directory'''
    for file in os.listdir(directory):
        strippedName = file[9:]
        os.rename(directory+"/"+file, directory+"/"+strippedName)

stripPrecedingCharsLS(sys.argv[1])