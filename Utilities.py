import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class DetectionUtilities:
    def __init__(self, xmlPathFace, xmlPathEyes):
        self.face_cascade = cv2.CascadeClassifier(xmlPathFace)
        self.eye_cascade = cv2.CascadeClassifier(xmlPathEyes)
    

    def GetFace(self, imgColor, imgGray, scale, minNeig, maxToCrop = 1, flags = 1):
        faceColor_crop = []
        faceGray_crop = []

        faceDetection = self.face_cascade.detectMultiScale(imgGray, scale, minNeig, flags)

        print('Can detect(face): ', len(faceDetection))

        for f in faceDetection:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(imgGray, (x, y), (x+w, y+h), (255, 0, 0), 4)
            cv2.rectangle(imgColor, (x, y), (x+w, y+h), (255, 0, 0), 4)
            """if len(faceDetection) == maxToCrop:
                cv2.imshow("Image", imgGray)
                cv2.waitKey(500)"""
            
        if len(faceDetection) == maxToCrop:
            faceGray_crop.append(imgGray[y:y+h, x:x+w])
            faceColor_crop.append(imgColor[y:y+h, x:x+w])

        return faceDetection, faceGray_crop, faceColor_crop

    def GetEyes(self, imgColor, imgGray, facesDetection, faceCrop = 1):

        faceGray_crop = []
        faceColor_crop = []
        eyesDetection = []

        for f in facesDetection:
            x, y, w, h = [ v for v in f ]

            faceGray = imgGray[y:y+h, x:x+w]
            faceColor = imgColor[y:y+h, x:x+w]

            eyesDetection = self.eye_cascade.detectMultiScale(faceGray) # Eyes detector
        
        print('Eyes detection: ', len(eyesDetection))

        if len(facesDetection) == faceCrop: 
            for (ex, ey ,ew, eh) in eyesDetection: 
                cv2.rectangle(faceGray, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 4)
                cv2.rectangle(faceColor, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 4)

            if len(eyesDetection) == 2:
                faceColor_crop.append(faceColor) # Facecolor
                faceGray_crop.append(faceGray) # Facegray

        return eyesDetection, faceGray_crop, faceColor_crop
    
    def ShowImage(self, img, eyes):
        
        fig= plt.figure(figsize=(5, 5))
        rows = 1

        fig.add_subplot(rows, 1, 1)
        plt.imshow(img[:,:,::-1])

        if len(eyes) == 2:
            for e in eyes:
                fig.add_subplot(rows, 2, 1) 
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.imshow(e[:,:,::-1])
                
        plt.show()

# Get crop face
def GetCropFaces(img, faces, maxToCrop = 1):
    face_crop = []

    try:
        for f in faces:
            x, y, w, h = [ v for v in f ]
            # Define the region of interest in the image  
            if len(faces) <= maxToCrop:
                face_crop.append(img[y:y+h, x:x+w])
                
        for face in face_crop:
            cv2.imshow('Face',face)
            cv2.waitKey(1000)
    except Exception as e:
        print('Exception: ', e)

    return face_crop

# Saving data

def SaveImage(img, total_facedetection, total_eyesDetection, num):
    datapath_saving = ''

    if total_eyesDetection == 2 and total_facedetection == 1:
        # Process finished correctly    
        datapath_saving = 'Data/Images/MeTest/Processing/' + 'out_' +str(num)+ '.jpg'
        print('Saving on: ', datapath_saving)
    elif total_eyesDetection > 2 or total_facedetection > 1:
        # Multiples faces on image
        datapath_saving = 'Data/Images/MeTest/MultipleFace/' + 'mult_' +str(num)+ '.jpg'
        print('Saving on: ', datapath_saving)
    else:
        # Face undetacteble
        datapath_saving = 'Data/Images/MeTest/Undetectable/' + 'undetect_' +str(num)+ '.jpg'
        print('Saving on: ', datapath_saving)

    SaveOpenCvImage(img, datapath_saving)

def SaveOpenCvImage(img, path):
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        print('Exception: ', e)

# SaveOpenCvImage(image, 'Data/Images/out.jpg')