import torchvision.transforms.functional as TF
import cv2
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
import os
import numpy as np
import xml.etree.ElementTree as ET

def calculateCountMetrics(predictedCounts, actualCounts, actualTotalInclAmbig = None):
    '''calculates count metrics for a single example'''

    predFluor = predictedCounts[0]
    predNonFluor = predictedCounts[1]
    predTotal = predFluor + predNonFluor

    actualFluor = actualCounts[0]
    actualNonFluor = actualCounts[1]
    if(actualTotalInclAmbig == None):
        actualTotal = actualFluor + actualNonFluor
    else:
        actualTotal = actualTotalInclAmbig

    try:
        predictedTransmission = (predFluor / predTotal) * 100
    except:
        predictedTransmission = 0

    actualTransmission = (actualFluor / actualTotal) * 100

    if(actualFluor != 0):
        fluorPercentageDiff = float((abs((predFluor)-(actualFluor) ) / (actualFluor) ) * 100)
    else:
        fluorPercentageDiff = predFluor   #think this over.... is this accurate??? not really.

    fluorKernelDiff = predFluor - actualFluor

            
    if(actualNonFluor != 0):
        nonFluorPercentageDiff = float((abs((predNonFluor)-(actualNonFluor) ) / (actualNonFluor) ) * 100)

    else:
        nonFluorPercentageDiff = predNonFluor  #think this over.... is this accurate??? not really.

    nonFluorKernelDiff = predNonFluor - actualNonFluor

    totalPercentageDiff = float((abs((predTotal)-(actualTotal) ) / (actualTotal) ) * 100)
    totalKernelDiff = predTotal - actualTotal

    if(actualTransmission != 0):
        transmissionPercentageDiff =  float((abs((predictedTransmission)-(actualTransmission) ) / (actualTransmission) ) * 100)
    else:
        transmissionPercentageDiff = predictedTransmission   #THIS IS BAD CHANGE IT


    #print("predictedTransmission: ", predictedTransmission, "   actualTransmission: ", actualTransmission)
    transmissionDiff = predictedTransmission - actualTransmission
    #print("transmission Diff: ", transmissionDiff)

    #what shall we return... 
    metricList = [fluorKernelDiff, fluorPercentageDiff, nonFluorKernelDiff, nonFluorPercentageDiff,  totalKernelDiff, totalPercentageDiff,  transmissionDiff, transmissionPercentageDiff]
    return metricList




def outputAnnotatedImg(imageTensor, annotations, name="outputImg.png"):
    img = TF.to_pil_image(imageTensor)
    imDraw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=25)


    labels = annotations["labels"]
    boxes = annotations["boxes"]

    classes = [None, "nonfluorescent", "fluorescent"]
    classColors = [None,(90,0,230),(175,255,0)] 

    for ind, label in enumerate(labels):
        #print(label, boxes[ind])
        box = boxes[ind]

        imDraw.text((box[0]+25, box[1]), classes[label], font=font,  fill=classColors[label])
        
        #Four points to define the bounding box. 
        '''
        coordinates = [(x1, y1), (x2, y2)]
        (x1, y1)
            *--------------
            |             |
            |             |
            |             |
            |             |
            |             |
            |             |
            --------------*
                        (x2, y2)
        '''
        rect = [(box[0], box[1]), (box[2],box[3])]
        imDraw.rectangle(rect, outline=classColors[label], width=3)
    img.save("OutputImages/"+name)


def findAmbiguousCalls(imageTensor, annotations, name):

    imgHeight = imageTensor.shape[1]
    imgWidth = imageTensor.shape[2]


    fluorCentroids  = np.zeros( ( imgHeight, imgWidth, 1), dtype="uint8")
    nonFluorCentroids = np.zeros((imgHeight, imgWidth, 1), dtype="uint8")
    ambiguousSpots = np.zeros((imgHeight, imgWidth, 1), dtype="uint8")

    labels = annotations["labels"]
    boxes = annotations["boxes"]
    
    for ind, label in enumerate(labels):

        box = boxes[ind]

        x1 = round(box[0].item())
        y1 = round(box[1].item())

        x2 = round(box[2].item())
        y2 = round(box[3].item())

        centroidX = round((x1+x2)/2)
        centroidY = round((y1+y2)/2)

        if(label==2):
            cv2.circle(fluorCentroids, (centroidX, centroidY), 8, (255,255,255), -1)
        elif(label==1):
            cv2.circle(nonFluorCentroids, (centroidX, centroidY), 8, (255,255,255), -1)

        
    ambiguousOverlaps = cv2.bitwise_and(fluorCentroids, nonFluorCentroids)
    #connectedComponentsWithStats inputs:   binaryImg, connectivity(4 or 8), outputimagelabelType (CV_32S  or CV_16U)
    #https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
    numberLabels, labelMatrix, stats, centroids = cv2.connectedComponentsWithStats(ambiguousOverlaps, 4, cv2.CV_32S )


    ambiguousCount = 0

    for i, c in enumerate(centroids):
        if(i>0):
            #print("stats:", stats[i])
            area = stats[i][4]
            if(area>20):
                cv2.circle(ambiguousSpots, (int(c[0]), int(c[1])), 8, (255,255,255), -1)
                ambiguousCount += 1
                #cv2.rectangle(ambiguousSpots, (int(stats[i][0]), int(stats[i][1])), (int(stats[i][0] + stats[i][2]), int(stats[i][1] + stats[i][3])), (255,255,255), 4)
    
    #cv2.imwrite(name.split(".")[0]+"_fluorDots.png", fluorCentroids)
    #cv2.imwrite(name.split(".")[0]+"_nonfluorDots.png", nonFluorCentroids)
    #cv2.imwrite(name.split(".")[0]+"_ambiguousOverlaps.png", ambiguousOverlaps)

    cv2.imwrite(name.split(".")[0]+"_ambiguousSpots.png", ambiguousSpots)

    return ambiguousCount


def outputAnnotatedImgCV(image, annotations, name="OutputImages/outputImg.png", tensor=True):
    #function to output visualized annotations on ear image using OpenCV instead of PIL to do the box drawing. 
    
    if(tensor):
        img = cv2.cvtColor(np.asarray(TF.to_pil_image(image)), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image, cv2.IMREAD_COLOR)

    labels = annotations["labels"]
    boxes = annotations["boxes"]

    classColors = [None,(255,50,195),(15, 255, 200)]

    for ind, label in enumerate(labels):
        #print(label, boxes[ind])
        box = boxes[ind]
    
        #Four points to define the bounding box. 
        '''
        coordinates = [(x1, y1), (x2, y2)]
        (x1, y1)
            *--------------
            |             |
            |             |
            |             |
            |             |
            |             |
            |             |
            --------------*
                        (x2, y2)
        '''
        x1 = round(box[0].item())
        y1 = round(box[1].item())

        x2 = round(box[2].item())
        y2 = round(box[3].item())

        rect = [(x1, y1), (x2,y2)]
        centroidX = round((x1+x2)/2)
        centroidY = round((y1+y2)/2)

        #cv2.rectangle(img, rect[0], rect[1], classColors[label], 4)
        cv2.circle(img, (centroidX, centroidY), 8, classColors[label], -1)

    cv2.imwrite(name, img)

 

def outputPointAnnotatedImg(image, annotationsXML, name="OutputImages/outputPointImg.png"):
    img = cv2.imread(image, cv2.IMREAD_COLOR)

    cv2.imwrite(name, img)

    print(annotationsXML)


    xmlTree = ET.parse(annotationsXML)
    xmlRoot = xmlTree.getroot()

    markerData =  xmlRoot.find('Marker_Data')

    classColors = [(15,255,200), (179,0,179), (255,255,255)]

    for markerType in markerData.findall("Marker_Type"):
        typeID = int(markerType.find('Type').text)
        if(typeID in [1,2,3]):
            for marker in markerType.findall("Marker"):
                xCoord = int(marker.find('MarkerX').text)
                yCoord = int(marker.find('MarkerY').text)

                cv2.circle(img, (xCoord, yCoord), 8, classColors[typeID-1], -1)
    cv2.imwrite(name, img)

def outputPredictionAsXML(prediction, outFileName):

    root = ET.Element("annotation")

    '''
    filename = ET.SubElement(root, "filename")
    filename.text = outFileName.split("/")[-1].split('.')[0]

    path = ET.SubElement(root, "path")
    path.text = outFileName
    
    '''


    #obj = ET.SubElement(root, "object")

    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']

    for ind, label in enumerate(labels):
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        bndbox = ET.SubElement(obj, "bndbox")
        confidence = ET.SubElement(obj, "score")
    
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")

        box = boxes[ind]
        score = scores[ind]
        if label.item() == 1:
            name.text = "nonfluorescent"
        elif label.item() == 2:
            name.text = "fluorescent"
            

        xmin.text = str(round(box[0].item()))
        ymin.text = str(round(box[1].item()))
        xmax.text = str(round(box[2].item()))
        ymax.text = str(round(box[3].item()))

        confidence.text  = str(round( score.item(), 5 ) )


    outFile = open(outFileName, 'wb')
    tree = ET.ElementTree(root)
    ET.indent(tree, '  ')
    tree.write(outFile)
    outFile.close()




def convertPVOC(annotationFile, imageSize):


    width = imageSize[0]
    height = imageSize[1]

    xmlTree = ET.parse(annotationFile)
    xmlRoot = xmlTree.getroot()

    boxes = []

    for obj in xmlRoot.findall('object'):
            xmin = int(float(obj.find('bndbox').find('xmin').text))
            xmax = int(float(obj.find('bndbox').find('xmax').text))
            ymin = int(float(obj.find('bndbox').find('ymin').text))
            ymax = int(float(obj.find('bndbox').find('ymax').text))

            x = xmin  #does LS want top left corner pixel, or center pixel??
            y = ymin

            boxW = xmax - xmin
            boxH = ymax - ymin

            label = obj.find('name').text

            boxInfo = [x,y,boxW,boxH, label]
            boxes.append(boxInfo)



    outputJson = open(annotationFile.split(".")[0]+"_LS.json", "w")

    outputJson.write("{ \n")
    outputJson.write('"data": {\n')
    outputJson.write('"image": '+ '"PLACEHOLDER"' + "\n},\n")

    outputJson.write('"annotations": '+ "[\n")
    outputJson.write("{\n")
    outputJson.write('"result": [\n')

    for i,b in enumerate(boxes):
        outputJson.write("{\n")
        outputJson.write('"original_width": ' + str(width) + ",\n")
        outputJson.write('"original_height": ' + str(height) + ",\n")
        outputJson.write('"image_rotation": ' + str(0) + ",\n")

        xPercent = b[0] / width * 100.0
        yPercent = b[1] / height * 100.0
        widthPercent =  b[2] / width * 100.0
        heightPercent = b[3] / height * 100.0

        outputJson.write('"value": {\n')
        outputJson.write('"x": ' + str(xPercent) + ",\n")
        outputJson.write('"y": ' + str(yPercent) + ",\n")
        outputJson.write('"width": ' + str(widthPercent) + ",\n")
        outputJson.write('"height": ' + str(heightPercent) + ",\n")
        outputJson.write('"rotation": ' + str(0) + ",\n")

        outputJson.write('"rectanglelabels": [\n' +'"'+ b[4] +'"'  + "]},\n")
        outputJson.write('"from_name": "label",\n "to_name": "image",\n "type": "rectanglelabels"\n')

        if(i<len(boxes)-1):
            outputJson.write("},\n")
        else:
            outputJson.write("}\n")

    outputJson.write("\n]\n")
    outputJson.write("}]\n")
    outputJson.write("}")
    outputJson.close()