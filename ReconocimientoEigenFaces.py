from asyncio import DatagramProtocol
from email.mime import image
import cv2
import os

dataPath = './DataInput'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# EigenFaces
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Leer el modelo.
face_recognizer.read('modeloEigenFace.xml')

# Entrada de video
cap = cv2.VideoCapture(0)

# Detector de rostros con HaarCascades
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Leer los fotogramas del video
while True:
    ret, frame = cap.read()
    if ret == False: break

    # Todo gris para procesar las imagenes.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_copy = gray.copy()

    # Detectamos el rostro. 
    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        rostro = gray_copy[y:y+h,x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        # predict() - Predice una etiqueta y la confianza (por ejemplo, la distancia), para una imagen de entrada asociada.
        result = face_recognizer.predict(rostro)

        # Texto con el nombre de la Persona y la coincidencia
        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        # Rectángulo para el rostro
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        # EigenFaces
        # Si el valor de coincidencia es mayor a "x" detecta como "Desconocido"
        if result[1] < 4200:
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
