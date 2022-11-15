import DuckDuckGoImages as ddg		#pip install DuckDuckGoImages


filtro = "carro"
destino = r"F:\Downloads\DIO\MACHINE LEARNING\Docs curso\Codigos\codigos do git\yolo\dataset\carros"

ddg.download('carros', max_urls=400, folder=destino,remove_folder=False, parallel=True)

#filtro, folder=destino, remove_folder=False, parallel=True)
#print('Iniciando downloads...')
#try:
#    ddg.download(filtro, folder=destino, remove_folder=False, parallel=True)
#except Exception as e:
#    print("type error: ", e)
#print('Downloads concluidos...')