import os
import face_recognition
import cv2
from deepface import DeepFace
import datetime

path = 'Images'
pathL = os.listdir(path)
images = []
Names = []
def createAttendanceSheet():
    for i in pathL:
        curImg = face_recognition.load_image_file(f'{path}/{i}')
        images.append(curImg)
        Names.append(os.path.splitext(i)[0])
    return Names,images
namesl,imagesl = createAttendanceSheet()
def getencoding(imagesl):
    encodel = []
    for img in imagesl:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]
        encodel.append(encodeimg)
    return encodel
encodelist = getencoding(imagesl)






