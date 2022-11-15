import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Transformando imagem colorida em tons de cinza

im = Image.open('cris.png') #Carrega imagem
pixels = im.load() #Carrega os valores rgb
grayImage = np.empty([im.size[1], im.size[0]], dtype=np.uint8) #Cria uma imagem "vazia" com o tamanho de altura e largura iguais ao da imagem lida

for i in range(im.size[1]):
    for j in range(im.size[0]):
        r = pixels[j,i][0] #Lê o valor R do pixel
        g = pixels[j,i][1] #Lê o valor G do pixel
        b = pixels[j,i][2] #Lê o valor B do pixel
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b #Formula padrão para converter rgb para escalas de cinza
        gray=int(gray) #Transforma o valor convertido em inteiro
        grayImage[i,j] = gray #Salva valor

result = plt.imshow(grayImage, cmap='gray', vmin=0, vmax=255) 
plt.savefig('res.png') #Salva imagem
plt.show() #Mostra a imagem

#Transformando a imagem em tons de cinza em preto/branca (valores binários)
binImage = np.empty([len(grayImage[1]),len(grayImage[0])], dtype=np.uint8) #Cria uma imagem "vazia" com o tamanho de altura e largura iguais ao da imagem lida

for i in range(len(grayImage[1])):
    for j in range(len(grayImage[0])):    
        if grayImage[i,j] < 128:
            binImage[i,j] = 1
        else:
            binImage[i,j] = 0

result2 = plt.imshow(binImage, cmap='binary', vmin=0, vmax=1) 
plt.savefig('res2.png') #Salva imagem
plt.show() #Mostra a imagem













