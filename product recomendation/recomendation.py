from random import random
import tensorflow as tf
from keras.applications import vgg16
from tensorflow.keras.utils import load_img, img_to_array

from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

imgs_path = "training\\"
imgs_model_width, imgs_model_height = 224, 224

nb_closest_images = 5 #Número de imagens mais próximas para serem mostradas ao fim do programa

#Carrega o modelo
vgg_model = vgg16.VGG16(weights='imagenet')
#Remove as duas últimas camadas para obter as features ao inves da predição da rede
feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
files = [imgs_path + x for x in os.listdir(imgs_path) if "jpg" in x]
print("Qtde de imagens:",len(files))

#Carrega a imagem original no formato PIL
original = load_img(files[0], target_size=(imgs_model_width, imgs_model_height))
plt.imshow(original)
plt.show()
print("image loaded successfully!")

#Converte a imagem para um array em formato Numpy
numpy_image = img_to_array(original)

#Converte as imagens p/ em fomrato batch
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)

#Prepara a imagem para o modelo VGG
processed_image = preprocess_input(image_batch.copy())

#Pega as features da imagemm
img_features = feat_extractor.predict(processed_image)

print("Numero de features:",img_features.size)
img_features

#Carrega todas as imagens p/ a rede load all the images and prepare them for feeding into the CNN
importedImages = []

for f in files:
    filename = f
    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    importedImages.append(image_batch)
    
images = np.vstack(importedImages)

processed_imgs = preprocess_input(images.copy())

#Extrai as features das imagens
imgs_features = feat_extractor.predict(processed_imgs)
print("Features extraidas.")
imgs_features.shape

#Pega a simiaridade (cosine_similarity) entre as imagens
cosSimilarities = cosine_similarity(imgs_features)

#Salva os resultados em um dataframe
cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)
cos_similarities_df.head()


#Função para retornar as imagens mais similares
def retrieve_most_similar_products(given_img):

    print("Foto original:")

    original = load_img(given_img, target_size=(imgs_model_width, imgs_model_height))
    plt.imshow(original)
    plt.show()

    print("Similares:")

    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index
    closest_imgs_scores = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1]

    for i in range(0,len(closest_imgs)):
        original = load_img(closest_imgs[i], target_size=(imgs_model_width, imgs_model_height))
        plt.imshow(original)
        plt.show()
        print("Similaridade: ",closest_imgs_scores[i])

#Chama a função de forma a escolher uma imagem randomica dentro das imagens disponíveis
retrieve_most_similar_products(files[int(random())])