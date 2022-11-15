import speech_recognition as sr
import playsound 
from gtts import gTTS, tts
import random
import webbrowser
import pyttsx3
import os
import requests

class virtualAssist():
    def __init__(self, assistName, person):
        self.assit_name = assistName
        self.person = person

        self.engine = pyttsx3.init() #Identificação do áudio

        self.r = sr.Recognizer() #Reconhecer a voz
        self.voice_data = '' #Armazena o texto do audio

    def engine_speak(self, text): 
        text = str(text)
        self.engine.say(text) #Fala da assistente (pyttsx3)
        self.engine.runAndWait()

    def record_audio(self, ask=""):
        #Microfone como fonte de audio:
        with sr.Microphone() as source:
            if ask:
                print('Inicio...')
                self.engine_speak(ask)

            audio = self.r.listen(source, 5, 5)
            print('Procurando')
            try:
                self.voice_data = self.r.recognize_google(audio, language="pt-BR") #Converte audio para texto

            except sr.UnknownValueError:
                self.engine_speak('Desculpe, não entendi')

            except sr.RequestError:
                self.engine_speak('Desculpa, não consegui me conectar ao servidor.') #Recognizer não está conectado

            print(">>",self.voice_data.lower()) #Printa o que foi falllado
            self.voice_data = self.voice_data.lower()

            return self.voice_data.lower()

    def engine_speak(self, audio_strig):
        audio_strig = str(audio_strig)
        tts = gTTS(text=audio_strig, lang='pt', tld='com.br') #Áudio em portugues
        r = random.randint(1,20000)
        audio_file = 'audio' + str(r) + '.mp3'
        tts.save(audio_file)
        playsound.playsound(audio_file)
        print(self.assit_name + ':', audio_strig)
        os.remove(audio_file)


    def there_exist(self, terms):
        #Função para identificar se o termo existe
        for term in terms:
            if term in self.voice_data:
                return True

#Respostas:
    def respond(self, voice_data):
        if self.there_exist(['oi', 'olá', 'opa', 'bão']):
            greetigns = f'Oi {self.person}, como você está?'
            self.engine_speak(greetigns)

        #Google
        if self.there_exist(['procure no google por']) and 'youtube' not in voice_data:
            word = voice_data.split("por")[-1]
            url =  "http://google.com/search?q=" + word
            webbrowser.get().open(url)
            self.engine_speak("Aqui está o que eu achei para " + word + 'no google')

        #Youtube
        if self.there_exist(["procure no youtube por"]):
            word  = voice_data.split("por")[-1]
            url = "http://www.youtube.com/results?search_query=" + word
            webbrowser.get().open(url)
            self.engine_speak("Aqui está o que eu achei para " + word + 'no youtube')

        #Criar pasta
        if self.there_exist(["crie uma pasta chamada"]):
            word  = voice_data.split("chamada")[-1]
            os.mkdir(word)
            self.engine_speak("Pasta " + word + "criada com sucesso.")

        #Clima:
        if self.there_exist(["como está o clima em"]):
            word = voice_data.split("em")[-1]
            url = 'https://wttr.in/{}'.format(word)
            res = requests.get(url)
            self.engine_speak("O clima em " + word + "será mostrado na tela")
            print(res.text)

        #Maps:
        if self.there_exist("qual é a rota de"):
            source = voice_data.split("de")[-1]
            list_of_words = source.split()
            source = list_of_words[0]
            destination = voice_data.split("para")[-1]
            url = "https://www.google.com.br/maps/dir/"+ source + "/" + destination
            webbrowser.get().open(url)
            self.engine_speak("Aqui está a rota de " + source + 'para' + destination)


assistent = virtualAssist('Nina', 'João')

while True:

    voice_data = assistent.record_audio('Ouvindo...')
    assistent.respond(voice_data)

    if assistent.there_exist(['tchau', 'adeus', 'tchauzinho', 'até mais']):
        assistent.engine_speak("Tchau, passar bem.")
        break