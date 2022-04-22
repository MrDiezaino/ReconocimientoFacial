import cv2
import os
import imutils

personName = 'Diego'
dataPath = './DataInput'
personPath = dataPath + "/" + personName

# Si no existe la carpeta la creamos
if not os.path.exists(personPath):
    print("Carpeta creada", personPath)
    os.makedirs(personPath)

# Captura del video de entrada
cap = cv2.VideoCapture(0)

# Detector de rostros con HaarCascades
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

# Leer los fotogramas del video
while True:
    ret, frame = cap.read()
    # Redimensionamos por si el tamaño es muy grande
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    
    # Filtro gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    auxFrame = frame.copy()

    # Detectamos el rostro.
    faces = faceClassif.detectMultiScale(gray,1.3,5)

    # Dibujar un rectángulo en el rostro y guardarlos con "x" tamaño
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count = count + 1
    
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    # Almacenamos 300 fotos de tu cara.
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()
