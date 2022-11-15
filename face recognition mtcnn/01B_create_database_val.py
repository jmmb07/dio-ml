from pickletools import optimize
from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray
import cv2


detector = MTCNN()

def extrair_face(arquivo, size=(160,160)):

    #img = Image.open(arquivo) #Caminho da figura
    #img = img.convert('RGB') #Converte em RGB
    img = cv2.cvtColor(cv2.imread(arquivo), cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img) #Resultado da face
    x1, y1, w, h = results[0]['box']
    x2, y2 = x1 + w, y1 + h
    array = asarray(img)
    face = array[y1:y2 , x1:x2]

    #Transformar imagem em quadrado:
    image = Image.fromarray(face)
    image = image.resize(size)

    return image

def carrega_fotos(sourceDir, targetDir):
    
    for filename in listdir(sourceDir):
        path = sourceDir + filename
        targetPath = targetDir + filename

        try:
            face = extrair_face(path)
            face.save(targetPath, 'JPEG', quality=100, optimize=True, progressive=True)
        except:
            print("Erro na imagem {}".format(path))

def carrega_diretorios(sourceDir, targetDir):

    for subdir in listdir(sourceDir):
        path = sourceDir + subdir + "\\"
        targetPath = targetDir + subdir + "\\"

        if not isdir(path):
            continue
    
        carrega_fotos(path, targetPath)


if __name__ == '__main__':
    carrega_diretorios("fotos_originais_val\\", "fotos_validation\\")

    #img = cv2.cvtColor(cv2.imread("F:\\Downloads\\DIO\\MACHINE LEARNING\\Docs curso\\Codigos\\face recognition do zero\\fotos originais\\nenem\\WhatsApp Image 2022-09-26 at 19.46.09 (2).jpeg"), cv2.COLOR_BGR2RGB)
    #a = detector.detect_faces(img)

    #x1, y1, w, h = a[0]['box']
    #x2, y2 = x1 + w, y1 + h
    #array = asarray(img)
    #face = array[y1:y2 , x1:x2]

    #size=(160,160)
   # image = Image.fromarray(face)
   # image = image.resize(size)
    #image.save("F:\\Downloads\\DIO\\MACHINE LEARNING\\Docs curso\\Codigos\\face recognition do zero\\fotos originais\\eu\\test", 'JPEG')

    #print("x1: ", x1, "y1: ", y1, "w: ", w, "h: ", h, "y2: ", y2, "x2: ", x2)
    #teste = extrair_face("F:\\Downloads\\DIO\\MACHINE LEARNING\\Docs curso\\Codigos\\face recognition do zero\\fotos originais\\eu\\WhatsApp Image 2022-09-26 at 20.10.35.jpeg")
    #cv2.imshow(teste)
    