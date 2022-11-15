import os

#salva todos os caminhos das imagens em um arquivo
with open("output.txt", "w") as a:
    for path, subdirs, files in os.walk(r'F:\Downloads\DIO\MACHINE LEARNING\Docs curso\Codigos\codigos do git\yolo\dataset'):
        for filename in files:
            if filename.endswith('.txt'):
                pass
            else:
                f = os.path.join(path, filename)
                a.write(str(f)+ os.linesep)
