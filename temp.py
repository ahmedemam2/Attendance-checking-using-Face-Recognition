import  face_recognition
import cv2

imgBarrack = face_recognition.load_image_file('Images/BarrackObama.jpg')
imgBarrack = face_recognition.face_encodings(imgBarrack)[0]
imgBarrack2 = face_recognition.load_image_file('Images/Unknown.jpg')
imgBarrack2 = face_recognition.face_encodings(imgBarrack2)[0]
results = face_recognition.compare_faces([imgBarrack], imgBarrack2)
distance = face_recognition.face_distance([imgBarrack],imgBarrack2)
print(results,distance)

