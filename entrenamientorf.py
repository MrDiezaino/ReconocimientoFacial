from xml.sax.handler import feature_namespaces
import cv2
import os
import numpy as np

dataPath = './DataInput'
# Ver toda la gente que está en el DataSet
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

# Leer las rutas de las imágenes de las personas
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo imágenes de: ' + nameDir)

    #Leer los rostros.
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        # Almacenar los rostros y las etiqueas
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName, 0))

        # Imagen en escala de grises
        image = cv2.imread(personPath+'/'+fileName, 0)

        # Preview de la lectura de imagenes 
        cv2.imshow('image', image)
        cv2.waitKey(10)

    label = label + 1

# El método EigenFaceRegoznicer necesita que las imágenes estén en escala de grises.
# además de asumir que todas las imágenes que leer tienen el mismo tamaño

# face_recognizer = cv2.face.EigenFaceRecognizer_create()

# FisherFace es una mejora de EigenFace
# face_recognizer = cv2.face.FisherFaceRecognizer_create()

# LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el recognizer
print("Entrenando...")

# Parámetros de "train" {
#   - Img de entrenamiento
#    - Las etiquetas de las imágenes (Deben ser np.arrays)
# }

face_recognizer.train(facesData, np.array(labels))

# Almacenar los modelos
# Puede ser xml o yaml

# face_recognizer.write('modeloEigenFace.xml')
# face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')

print("Modelo almacenado")

cv2.destroyAllWindows()

