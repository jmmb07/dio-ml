from calendar import c
import numpy as np
from numpy import asarray, expand_dims
from PIL import Image
from mtcnn import MTCNN
import tensorflow as tf
from keras.models import load_model
import cv2


pessoa = ["cris", "messi"]
num_classes = len(pessoa)


detector = MTCNN() #Aonde está a face
facenet = load_model("facenet_keras.h5") #Transformar faces em embbeds
model = load_model("faces.h5") #Classificação das faces

def extrair_face(image, box, required_size=(160,160)):
    pixels = np.asarray(image)
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 +h

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)

    return np.asarray(image)

def get_embedding(model, face_pixels):
    
    face_pixels = face_pixels.astype('float32')
    #Padronizar cada imagem
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/std

    #Transformar a face em 1 unico exemplo:
    samples = expand_dims(face_pixels, axis=0) #(160, 160) -> (1, 160, 160)

    #realizar a predição gerando o embedding
    yhat = model.predict(samples)
    return yhat[0]

#Para uma única foto:

arquivo = "cristiano e messi.jpeg"
img = cv2.imread(arquivo)

faces = detector.detect_faces(img)

for face in faces:
    confidence = face['confidence'] #nivel de confianca da mtcnn
    strconf = str("{:.2f}".format(confidence*100))
    if confidence >=0.7:
        x1, y1, w, h = face['box']
        face = extrair_face(img, face['box'])
        face = face.astype("float32")/255
        emb = get_embedding(facenet, face)
        tensor = np.expand_dims(emb, axis=0)
    
        prob = model.predict(tensor)[0]
        classe = np.argmax(prob, axis=0)
        user =  str(pessoa[classe]).upper()

        if classe == 1: 
            color = (143, 79, 27) #BGR
        else:
            color= (0, 0, 255) #BGR

        cv2.rectangle(img, (x1,y1), (x1+w, y1+h), color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        cv2.putText(img, user + " - " + strconf + "%", (x1, y1-10), font, fontScale=font_scale, color=color, thickness=2)

cv2.imwrite("Resultado.jpeg", img)
cv2.imshow("FACE TEST", img)
key = cv2.waitKey(0)

if key == 27: #ESC
   cv2.destroyAllWindows()



##Para capturar video:
#cap = cv2.VideoCapture("meci e cris.mp4")
#while True: 
#    _, frame = cap.read()    
#    faces = detector.detect_faces(frame)
#    for face in faces:
#        confidence = face['confidence'] #nivel de confianca da mtcnn
#        strconf = str("{:.2f}".format(confidence*100))
#        if confidence >=0.7:
#            x1, y1, w, h = face['box']
#            face = extrair_face(frame, face['box'])
#            face = face.astype("float32")/255
#            emb = get_embedding(facenet, face)
#            tensor = np.expand_dims(emb, axis=0)
#         
#            prob = model.predict(tensor)[0]
#            classe = np.argmax(prob, axis=0)
#            user =  str(pessoa[classe]).upper()
#            
#            if classe == 1: 
#                color = (143, 79, 27) #BGR
#            else:
#                color= (0, 0, 255) #BGR
#            
#            cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), color, 2)
#            font = cv2.FONT_HERSHEY_DUPLEX
#            font_scale = 0.5
#            cv2.putText(frame, user + " - " + strconf + "%", (x1, y1-10), font, fontScale=font_scale, color=color, thickness=2)
#
#    cv2.imshow("FACE TEST", frame)
#    key = cv2.waitKey(1)
#    if key == 27: #ESC
#        break
#
#cap.release()
#cv2.destroyAllWindows()


