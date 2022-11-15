#Treinamento de uma rede utilizando o conhecimento de uma rede já existente. 
#Baseado em: https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb
#Link do Colab: https://colab.research.google.com/drive/17TzVG1Hq5TtufClMqpS4TytyJ5NbXrEh?usp=sharing


#importando bibliotecas
import os
import random
import numpy as np
import keras

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.utils import load_img, img_to_array
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

root = '/dataset' #caminho das pastas com imagem
exclude = ['.ipynb_checkpoints'] #excluir categoria caso necessario
train_split, val_split = 0.7, 0.15 #dividindo o dataset em imagens de treinamento e validação

#dividindo dataset em categorias de acordo com a pasta do dataset
categories = [x[0] for x in os.walk(root) if x[0]][1:] 
categories = [c for c in categories if c not in [os.path.join(root, e) for e in exclude]]

print(categories) #imprimindo categorias (gatos e cachorros)

#funcao auxiliar para carregar as imagens corretamente no array, além de fazer um re-size para 224x224 pixels 
#para ser utilizadas na rede usada para o transfer learning posteriormente (vgg16)
def get_image(path):
    img = load_img(path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

#carregando todas as imagens do dataset 
data = []
for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames 
              in os.walk(category) for f in filenames 
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    for img_path in images:
        img, x = get_image(img_path)
        data.append({'x':np.array(x[0]), 'y':c})

# contando numero de classes
num_classes = len(categories)

random.shuffle(data) #coloca o dataset em ordem aleatoria

#separando as imagens em categorias: terinamento (70% das imagens) / validação (15% das imagens) / teste (15% das imagens)
idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

#separando os dados em rótulos
x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]
#print(y_test)

#normalizando os dados
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#convertendo rotulos em vetores onehot
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_val = keras.utils.np_utils.to_categorical(y_val, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
#print(y_test.shape)

print("finished loading %d images from %d categories"%(len(data), num_classes))
print("train / validation / test split: %d, %d, %d"%(len(x_train), len(x_val), len(x_test)))
print("training data shape: ", x_train.shape)
print("training labels shape: ", y_train.shape)

#carrega a rede VGG16 
vgg = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
vgg.summary()

#faz referencia a camada de entrada rede VGG16
inp = vgg.input

#cria uma nova camada para substituir a ultima camada da rede VGG16
new_classification_layer = Dense(num_classes, activation='sigmoid')

#conecta a nova camada na penultima camada da VGG
out = new_classification_layer(vgg.layers[-2].output)

#cria uma nova rede entre "inp" e "out"
model_new = Model(inp, out)

#para aproveitar a rede VGG, treinaremos somente a última camada que criamos deixando todas as camadas (exceto a última) "não treináveis"
for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False

#deixa a última camada (que criamos) "treinavel"
for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

model_new.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_new.summary()


#treinando a rede com o dataset de cachorros e gatos com 10 épocas
treino = model_new.fit(x_train, y_train, 
                         batch_size=128, 
                         epochs=10, 
                         validation_data=(x_val, y_val))


#mostrando o resultado em gráficos de 
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(treino.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epocas")

ax2 = fig.add_subplot(122)
ax2.plot(treino.history["val_accuracy"])
ax2.set_title("precisão de validação")
ax2.set_xlabel("epocas")
ax2.set_ylim(0, 1)

plt.show()