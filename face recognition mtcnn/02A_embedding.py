from hashlib import new
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray, expand_dims
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

#Transforma imagem em array
def carrega_face(arquivo):
    
    image = Image.open(arquivo)
    image = image.convert("RGB")
    
    return asarray(image)

#Carregar todas as imagens de um diretório
def carregar_all_faces(sourceDir):

    faces = list()
    for filename in listdir(sourceDir):

        path = sourceDir + filename
        try:
            img = carrega_face(path)
            faces.append(img)
        except: 
            print("Erro na imagem {}".format(path))
    
    return faces

def carrega_all_fotos(sourceDir):
    #em X = vetores das fotos das duas classes e em y = labels
    X, y = list(), list()

    for subdir in listdir(sourceDir):
        path = sourceDir + subdir + "\\"
        
        if not isdir(path):
            continue

        faces = carregar_all_faces(path)
        labels = [subdir for _ in range(len(faces))]

    #Saber onde o programa esta:
        print("Carregadas %d faces da classe %s " % (len(faces), subdir))

        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y) #retorna como numpy

def get_embedding(model, face_pixels):
    
    #Padronizar cada imagem
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/std

    #Transformar a face em 1 unico exemplo:
    samples = expand_dims(face_pixels, axis=0) #(160, 160) -> (1, 160, 160)

    #realizar a predição gerando o embedding
    yhat = model.predict(samples)
    return yhat[0]

if __name__ == '__main__':
    trainX, trainy = carrega_all_fotos("fotos somente face\\")
    print(trainy)
    #print(trainX.shape, trainy.shape)

    model = tf.keras.models.load_model("facenet_keras.h5")

    newTrainX = list()
    for face in trainX:
        embedding = get_embedding(model, face)
        newTrainX.append(embedding)

    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)

    df = pd.DataFrame(data=newTrainX)
    df['target'] = trainy
    #print(df)
    df.to_csv('faces.csv', index=False)

