from ast import increment_lineno
from codecs import utf_16_be_decode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.utils import to_categorical
from keras import models, layers
#%matplotlib inline

def print_confusion_matrix(model, valY, yhat_val):

    cm = confusion_matrix(valY, yhat_val)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1])/total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print("Modelo: {}".format(model))
    print("Acurácia: {:.4f}".format(acc))
    print("Sensitividade: {:.4f}".format(sensitivity))
    print("Especificidade: {:.4f}".format(specificity))

    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat = cm, figsize=(5, 5))
    plt.show()

#Carregar faces
df = pd.read_csv("faces.csv")
X = np.array(df.drop('target', axis=1)) #retirando o 'target' do dataframe
y = np.array(df.target)

#Misturar os dados:
trainX, trainY = shuffle(X, y, random_state=42)
#Binarizar/Discretizar as labels:
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)


#Carregar faces de validacao:
df_val = pd.read_csv("faces_validation.csv")
valX = np.array(df_val.drop('target', axis=1)) #retirando o 'target' do dataframe
valY = y_val = np.array(df_val.target)

#Binarizar/Discretizar as labels:
out_encoder.fit(valY)
valY = out_encoder.transform(valY)


#Utilizando o algoritmo Keras (multi layer perception)
trainY = to_categorical(trainY) #Categoriza os labels
print(trainY)
valY = to_categorical(valY)     #Categoriza os labels
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(128,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=100, batch_size=8)

yhat_train = model.predict(trainX)
yhat_val = model.predict(valX) #Retorna a probabilidade de ser a classe

#"Descategorizando" as labels de volta:
yhat_val = np.argmax(yhat_val, axis=1) #Dessa forma, yhat_val retornará 0 ou 1 
valY = np.argmax(valY, axis=1)

print_confusion_matrix("KERAS", valY, yhat_val)

model.save("faces.h5")