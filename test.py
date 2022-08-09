import main
from main import cv2 , face_recognition, np
encodelist,namesl = main.encodelist, main.namesl
cap = cv2.VideoCapture('Videos/1.mp4')
while True:
    state,frame = cap.read()
    frameres = cv2.resize(frame,(0,0),None,0.25,0.25)
    frameres = cv2.cvtColor(frameres,cv2.COLOR_BGR2RGB)
    facesloc = face_recognition.face_locations(frameres)
    encodeframe = face_recognition.face_encodings(frameres,facesloc)

    for encface, faceloc in zip(encodeframe,facesloc):
        isMatch = face_recognition.compare_faces(encodelist,encface)
        facedistance = face_recognition.face_distance(encodelist,encface)
        matchindex = np.argmin(facedistance)
        if isMatch[matchindex]:
            name = namesl[matchindex]
            print(name)
            cv2.rectangle(frame, (faceloc[3]*4, faceloc[0]*4), (faceloc[1]*4, faceloc[2]*4), (0, 255, 0), 5)
            cv2.putText(frame, name, (faceloc[3]*4, faceloc[0]*4 - 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
    cv2.imshow('Live Detection',frame)
    cv2.waitKey(1)
