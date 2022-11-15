O código inteiro pode ser separado em 4 partes:

1) [create_database.py](https://github.com/jmmb07/dio-ml/blob/main/face%20recognition%20mtcnn/01A_create_database.py): 
Esse código é responsável por criar a database da forma correta. O código carrega as imagens dos diretório "fotos originais", detecta as faces nas imagens e as salva no diretório "fotos somente face" com o tamanho pre-definido de 160x160. 
Obs.: é importante que as fotos originais tenham somente a pessoa para qual a rede deve ser treinada.

2) [embedding.py](https://github.com/jmmb07/dio-ml/blob/main/face%20recognition%20mtcnn/02A_embedding.py):
Gera os "embedds" para cada classe.

3) [validation_test.py](https://github.com/jmmb07/dio-ml/blob/main/face%20recognition%20mtcnn/03validation_testing.py);
Valida a rede e a salva.

4) [test.py](https://github.com/jmmb07/dio-ml/blob/main/face%20recognition%20mtcnn/04test.py):
Código que carrega o modelo e uma imagem para gerar o resultado.
