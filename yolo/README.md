# Instalar a darknet seguindo os passos descritos: https://github.com/AlexeyAB/darknet

# Criar dataset:

    1.1) Baixar imagens;
    
    1.2) Instalar o labelimg;
    
    1.3) Anotar as imagens:    
        1.3.1) Criar um txt com os labels das imagens;
        
        1.3.2) Abrir o labelimg: labelImg <"caminho das imagens"> <"caminho do label.txt">;
        
        1.3.3) Criar RectBox em cada imagem;
        
    1.4) Splitar as imagens em train e em test (arquivos de texto: train.txt e test.txt).

# Configurar o arquivos para YOLO (para yolo)
    2.1) Copiar os arquivos da darknet para o seu workspace e renomea-los:
        ./darknet/cfg/coco.data -> arquivo com caminho de arquivos de imagens, etc;
        ./darknet/cfg/yolov4-custom.cfg -> configurar de acordo com o github da darknet

# Treinar:
    3.1) Baixar arquivo com pesos (yolov4.conv.137);
    3.2) Copiar arquivo de pesos e o binário da darknet para a pasta de trabalho;
    3.3) Treinar com o seguinte comando: .\darknet.exe detector train carro-moto.data carro-moto.cfg yolov4.conv.137

# Resultados
    4.1) Ao treinar, a rede gera um gráfico mostrando a queda na % de loss;
    4.2) Executar comando para testar a rede: .\darknet.exe detector test carro-moto.data carro-moto.cfg backup\carro-moto_5000.weights (Obs: a darknet salva pesos a cada 1000 iterações e nem sempre o último arquivo gera o mellhor resultado). Após carregar a rede, entre com a imagem para testar.

Grafico de loss gerado:

<img src="https://user-images.githubusercontent.com/65690581/201943214-547c212c-4bf5-469a-98c5-f360d5da5188.png" width="600">



Resultado é salvo:

![predictions](https://user-images.githubusercontent.com/65690581/201822687-40fdf7f0-a237-4306-8857-09b9755742ac.jpg)


Log do resultado: 

![results](https://user-images.githubusercontent.com/65690581/201822639-0ead18aa-ed78-4801-81b4-d27cc8da794f.png)


    
