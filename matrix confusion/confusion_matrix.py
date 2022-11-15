import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data() #Criando dataset

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0

classes=[0,1,2,3,4,5,6,7,8,9]

#Modelando a rede
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_images, 
            y=train_labels, 
            epochs=1, 
            validation_data=(test_images, test_labels))


y_true=test_labels
y_pred=np.argmax(model.predict(test_images), axis=-1)

#Calculo dos requisitos
cf = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
rec =  recall_score(y_true, y_pred, average=None)
prec = precision_score(y_true, y_pred, average=None)

print("Matriz de confusão: \n", cf)
print("\nAcurácia da rede: ", acc)

fscore = dict()
print ("\nSensibilidade e precisão para cada classe:")
for i in range (0, len(classes)):
    print("\nClasse", i,"->")
    print("Sensibilidade = ", rec[i])
    print("Precisão = ", prec[i])
    fscore[i] = 2*(prec[i]*rec[i]/(prec[i]+rec[i]))
    print("F-Score = ", fscore[i])