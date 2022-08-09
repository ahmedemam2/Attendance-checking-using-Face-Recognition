import main
from main import cv2 , face_recognition , DeepFace
import datetime

def CheckAttendance(name,state):
    with open('Attendants.csv','r+') as f:
        AttendantsList = f.readlines()
        namelist = []
        print(AttendantsList)
        for line in AttendantsList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.datetime.now()
            DaTime = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{DaTime},{state}')



states = []
encodelist,namesl = main.encodelist, main.namesl
cap = cv2.VideoCapture(0)
while True:
    state,frame = cap.read()
    frameres = cv2.resize(frame,(0,0),None,0.25,0.25)
    frameres = frame
    frameres = cv2.cvtColor(frameres,cv2.COLOR_BGR2RGB)
    facesloc = face_recognition.face_locations(frameres)
    encodeframe = face_recognition.face_encodings(frameres,facesloc)
    # zip to iterate on both in same loop
    for encface, faceloc in zip(encodeframe,facesloc):
        isMatch = face_recognition.compare_faces(encodelist,encface)
        facedistance = face_recognition.face_distance(encodelist,encface)
        ind = isMatch.index(True)
        for match in isMatch:
            if match == True:

                name = namesl[ind]
                predict = DeepFace.analyze(frameres)
                states=predict['dominant_emotion']
                CheckAttendance(name,states)

    cv2.imshow('Live Detection',frame)
    cv2.waitKey(1)
