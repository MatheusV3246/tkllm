import sounddevice as sd
import scipy.io.wavfile as wavfile
from gtts import gTTS
from pynput import keyboard
import numpy as np
import os
from dotenv import load_dotenv
import pygame
from langchain_groq import ChatGroq
from faster_whisper import WhisperModel
import asyncio
import threading

# Carregar o modelo Whisper para transcrição
whisper_model = WhisperModel("small", 
                                compute_type="int8", 
                                cpu_threads=os.cpu_count(), 
                                num_workers=os.cpu_count())

load_dotenv()  # Carregar variáveis de ambiente do arquivo .env

class GravadorDeVoz:
    def __init__(self, taxa_amostragem=16000, 
                 pasta_audios='audios'):
        self.taxa_amostragem = taxa_amostragem
        self.pasta_audios = pasta_audios  # Pasta para armazenar os arquivos de áudio
        os.makedirs(self.pasta_audios, exist_ok=True)  # Criar a pasta se não existir
        self.arquivo_path = os.path.join(self.pasta_audios, 'gravacao.wav')  # Caminho do arquivo de gravação
        self.esta_gravando = False
        self.dados_audio = []
        self.esta_reproduzindo = False  # Variável para controlar a reprodução do áudio
        self.llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192")  # Configuração do modelo de linguagem

    def iniciar_gravacao(self):
        """Iniciar a captura de voz."""
        print("Capturando voz...")
        self.dados_audio = []  # Reiniciar os dados de áudio
        self.stream = sd.InputStream(samplerate=self.taxa_amostragem, channels=1, callback=self.callback_audio)
        self.stream.start()

    def parar_gravacao(self):
        """Parar a captura de voz e salvar o áudio gravado."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.dados_audio:  # Somente grave se houver dados
            audio_np = np.concatenate(self.dados_audio, axis=0)
            wavfile.write(self.arquivo_path, self.taxa_amostragem, audio_np)  # Salvar como arquivo WAV

    def callback_audio(self, indata, frames, time, status):
        """Callback chamado durante a captura de áudio."""
        if status:
            return  # Ignorar erros de status
        self.dados_audio.append(indata.copy())  # Adicionar o trecho gravado aos dados de áudio

    def transcrever_audio(self):
        """Transcrever o áudio gravado para texto."""
        try:
            segmentos, _ = whisper_model.transcribe(self.arquivo_path, language="pt")
            return "".join(segment.text for segment in segmentos).strip()  # Retornar a transcrição
        except Exception:
            return ""  # Silenciar erros de transcrição

    def ao_pressionar(self, key):
        """Função chamada ao pressionar uma tecla."""
        try:
            if key.char == 'r':  # Iniciar ou parar a gravação
                self.trocar_gravacao()
            elif key.char == 'c' and self.esta_reproduzindo:  # Cancelar a reprodução
                self.cancelar_reproduzindo()
            elif key.char == 'n':  # Nova pergunta
                self.iniciar_nova_pergunta()
        except AttributeError:
            pass  # Ignorar teclas especiais como shift, ctrl, etc.

    def trocar_gravacao(self):
        """Alternar entre iniciar e parar a gravação."""
        if not self.esta_gravando:
            self.esta_gravando = True
            self.iniciar_gravacao()
        else:
            print("Gravação finalizada!")
            self.esta_gravando = False
            self.parar_gravacao()
            transcricao = self.transcrever_audio()
            if transcricao:  # Apenas prossegue se a transcrição foi bem-sucedida
                print("\nUsuário:", transcricao)

                threading.Thread(target=self.processar_resposta, args=(transcricao,)).start()

    def processar_resposta(self, transcricao):
        """Processar a resposta da LLM em uma thread separada."""
        resposta_llm = self.llm.invoke(f"Responda de forma curta e objetiva: {transcricao}").content
        print("LLM:", resposta_llm, "\n")
        asyncio.run(self.falar(resposta_llm))  # Chama a função de fala de forma assíncrona

    def cancelar_reproduzindo(self):
        """Cancelar a reprodução do áudio."""
        print("Reprodução cancelada.")
        pygame.mixer.music.stop()  # Cancela a reprodução
        self.esta_reproduzindo = False  # Atualiza o estado de reprodução
        print("Você pode pressionar 'r' para ouvir a resposta novamente ou 'n' para fazer uma nova pergunta.")

    async def falar(self, texto):
        """Converter texto em fala e reproduzir o áudio gerado."""
        if not self.esta_reproduzindo:  # Certifica-se de que não há outro áudio sendo reproduzido
            arquivo_saida = self.texto_para_fala(texto)  # Cria um novo arquivo para cada resposta
            if arquivo_saida:
                await self.reproduzir_audio(arquivo_saida)  # Reproduz o novo arquivo
                self.limpar_arquivos()  # Limpa arquivos não utilizados após a reprodução
        else:
            print("Outro áudio está sendo reproduzido. Aguarde o término para ouvir o próximo.")

    def texto_para_fala(self, texto):
        """Converter texto em um arquivo de áudio."""
        arquivo_saida = os.path.join(self.pasta_audios, f"saida_{np.random.randint(100000)}.mp3")  # Gera um nome de arquivo aleatório
        try:
            tts = gTTS(text=texto, lang='pt')
            tts.save(arquivo_saida)  # Salvar o arquivo de áudio
            return arquivo_saida
        except Exception:
            return ""  # Silenciar erros na conversão

    async def reproduzir_audio(self, arquivo_path):
        """Reproduzir o arquivo de áudio gerado."""
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(arquivo_path)
            print("Reproduzindo áudio...")
            pygame.mixer.music.play()
            self.esta_reproduzindo = True  # Atualiza o estado de reprodução
            
            # Loop para verificar se a música está tocando
            while self.esta_reproduzindo:
                if not pygame.mixer.music.get_busy():
                    print("Áudio finalizado.")
                    self.esta_reproduzindo = False  # Atualiza o estado de reprodução
                    print("Você pode fazer uma nova pergunta agora.\n")  # Opção de iniciar nova pergunta
                # Permitir a interrupção da reprodução
                for evento in pygame.event.get():
                    if evento.type == pygame.KEYDOWN:
                        if evento.key == pygame.K_c:  # Permite parar a reprodução com 'c'
                            self.cancelar_reproduzindo()

        except Exception:
            pass  # Silenciar erros na reprodução

    def limpar_arquivos(self):
        """Remover arquivos de áudio não utilizados."""
        for arquivo in os.listdir(self.pasta_audios):
            caminho_arquivo = os.path.join(self.pasta_audios, arquivo)
            # Adicione sua lógica para determinar se o arquivo é "não utilizado"
            if arquivo.endswith('.mp3') and "saida" in arquivo:
                try:
                    os.remove(caminho_arquivo)  # Limpar o arquivo temporário
                    print("")
                except Exception:
                    print("")

    def iniciar_nova_pergunta(self):
        """Iniciar uma nova pergunta."""
        if not self.esta_reproduzindo:  # Verifica se não está reproduzindo
            print("Iniciando nova pergunta...")
            self.trocar_gravacao()  # Reinicia a gravação

    def iniciar(self):
        """Iniciar o loop principal do gravador de voz."""
        print("Aperte 'r' para começar/parar a captura de voz...")
        print("Aperte 'c' para cancelar a reprodução da resposta...")
        print("Aperte 'n' para fazer uma nova pergunta...")
        with keyboard.Listener(on_press=self.ao_pressionar) as listener:
            listener.join()

# Uso do exemplo:
gravador_de_voz = GravadorDeVoz()  # No duration needed, as we stop on second keypress
gravador_de_voz.iniciar()  # Press 'r' to start/stop recording
