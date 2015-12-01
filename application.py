#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import StringVar
from tkinter import messagebox

from file_manager import *

class Application(tk.Frame):
    """docstring for Application"""

    def __init__(self, master=None):
        """Classe que implementa a interface grafica da aplicacao Multilayer Perceptron"""

        #Inicializa o Frame
        tk.Frame.__init__(self, master)

        # Modifica o titulo da janela
        master.title('Multilayer Perceptron Algorithm')

        # Seta a organizacao da janela do tipo Grid
        self.grid()

        # Chama a funcao para criar os botoes
        self.create_buttons()

    def create_buttons(self):
        """Funcao que cria os botoes no Frame"""

        # Cria o botao abrir arquivo de treinamento
        tk.Button(self, text='Abrir Arquivo de Treinamento...', command=self.open_file(0)).grid(column = 0, row = 0)

        # Cria o botao para remover atributos da tabela
        tk.Button(self, text='Abrir Arquivo de Teste...', command=self.open_file(1)).grid(column = 1, row = 0)

        # Cria o botao executar id3
        tk.Button(self, text='Treinar Rede Neural...', command=self.execute).grid(column = 2, row = 0)

        # Cria o botao executar id3
        tk.Button(self, text='Testar Rede Neural...', command=self.execute).grid(column = 3, row = 0)


    def open_file(self,type):
        """Abre um File Dialog que retorna o nome do arquivo"""

        # Define as opcoes para abrir um arquivo
        self.file_opt = options = {}
        options['defaultextension'] = '.csv'
        options['filetypes'] = [('all files', '.*'), ('csv files', '.csv')]
        options['initialdir'] = 'C:\\'
        options['parent'] = root
        options['title'] = 'Escolha o arquivo de entrada'

        # Abre o FileDialog e recebe o nome do arquivo escolhido
        filename = filedialog.askopenfilename(**self.file_opt)
        name = os.path.split(filename)[1]

        # Verifica se o arquivo foi escolhido
        if filename:
            fm = FileManager()

            if(type == 0): # Verifica se o arquivo escolhido e o de treinamente
                # Le os dados de entrada a partir de um arquivo csv
                file_content_trainning = fm.read_csv(filename)
                # Cria a label com o nome do arquivo carregado
                tk.Label(self, text=name).grid(column=0,row=1)
            elif(type == 1): #Verifica se o arquivo escolhido e o de teste
                # Le os dados de entrada a partir de um arquivo csv
                file_content_testing = fm.read_csv(filename)
                # Cria a label com o nome do arquivo carregado
                tk.Label(self, text=name).grid(column=1,row=1)

    def execute(self):
        """Funcao que executa o algoritmo do Multilayer Perceptron"""

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
