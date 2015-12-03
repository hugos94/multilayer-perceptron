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

        # Define as opcoes para abrir um arquivo
        self.file_opt = options = {}
        options['defaultextension'] = '.csv'
        options['filetypes'] = [('all files', '.*'), ('csv files', '.csv')]
        options['initialdir'] = 'C:\\'
        options['parent'] = root
        options['title'] = 'Escolha o arquivo de entrada'

    def create_buttons(self):
        """Funcao que cria os botoes no Frame"""

        # Cria o botao para abrir o arquivo de treinamento
        tk.Button(self, text='Abrir Arquivo de Treinamento...', command=self.open_file_trainning).grid(column = 0, row = 0)

        # Cria o botao para inserir a quantidade de epocas do treinamento
        tk.Button(self, text="Quantidade de Épocas", command=self.get_epoch).grid(column=1,row=0)

        # Cria o botao para inserir a taxa de aprendizagem do treinamento
        tk.Button(self, text="Taxa de Aprendizagem", command=self.get_learning_tax).grid(column=2,row=0)

        # Cria o botao para treinar a MLP
        tk.Button(self, text='Treinar Rede Neural...', command=self.trainning_mlp).grid(column = 3, row = 0)

        # Cria o botao para abrir o arquivo de teste
        tk.Button(self, text='Abrir Arquivo de Teste...', command=self.open_file_test).grid(column = 4, row = 0)

        # Cria o botao para testar a MLP
        tk.Button(self, text='Testar Rede Neural...', command=self.test_mlp).grid(column = 5, row = 0)


    def open_file_trainning(self):
        """Abre um File Dialog que retorna o nome do arquivo de treinamento"""

        # Abre o FileDialog e recebe o nome do arquivo escolhido
        filename = filedialog.askopenfilename(**self.file_opt)
        name = os.path.split(filename)[1]

        # Verifica se o arquivo foi escolhido
        if filename:
            fm = FileManager()
            # Le os dados de entrada a partir de um arquivo csv
            self.file_content_trainning = fm.read_csv(filename)
            # Cria a label com o nome do arquivo carregado
            tk.Label(self, text=name).grid(column=0,row=1)

    def open_file_test(self):
        """Abre um File Dialog que retorna o nome do arquivo de teste"""

        # Abre o FileDialog e recebe o nome do arquivo escolhido
        filename = filedialog.askopenfilename(**self.file_opt)
        name = os.path.split(filename)[1]

        # Verifica se o arquivo foi escolhido
        if filename:
            fm = FileManager()
            # Le os dados de entrada a partir de um arquivo csv
            self.file_content_testing = fm.read_csv(filename)
            # Cria a label com o nome do arquivo carregado
            tk.Label(self, text=name).grid(column=4,row=1)

    def get_epoch(self):
        """ Cria uma nova janela pra receber a quantidade de epocas para o treinamento. """

        self.window = tk.Toplevel(self)
        self.window.title("Quantidade de Epocas!")
        self.window.grid

        tk.Label(self.window, text="Informe a quantidade de épocas:").grid(column=0,row=0)

        self.epoch = StringVar()
        tk.Entry(self.window,width=20, textvariable=self.epoch).grid(column=1,row=0)

        tk.Button(self.window,text="Salvar!", command=self.verify_entry_epoch).grid(column=2,row=0)

    def verify_entry_epoch(self):
        """ Metodo que verifica se a quantidade de epocas foi inserida. """
        if self.epoch.get():
            tk.Label(self,text=self.epoch.get()).grid(column=1,row=1)
            self.window.destroy()

    def get_learning_tax(self):
        """ Cria uma nova janela pra receber a taxa de aprendizagem para o treinamento. """
        self.window = tk.Toplevel(self)
        self.window.title("Taxa de Aprendizagem!")
        self.window.grid

        tk.Label(self.window, text="Informe a taxa de aprendizagem:").grid(column=0,row=0)

        self.learning_tax = StringVar()
        tk.Entry(self.window,width=20, textvariable=self.learning_tax).grid(column=1,row=0)

        tk.Button(self.window,text="Salvar!", command=self.verify_entry_learning_tax).grid(column=2,row=0)

    def verify_entry_learning_tax(self):
        """ Metodo que verifica se a taca de aprendizagem foi inserida. """
        if self.learning_tax.get():
            tk.Label(self,text=self.learning_tax.get()).grid(column=2,row=1)
            self.window.destroy()

    def trainning_mlp(self):
        """ Funcao que treina o algoritmo do Multilayer Perceptron. """
        #if (self.file_content_trainning and self.learning_tax.get() and self.epoch.get()):
        #    print("Deu certo!")
        #else:
        #    tk.messagebox.showwarning("Nenhum atributo escolhido", "Escolha um atributo para continuar!")
        #print(self.file_content_trainning)
        #print(self.learning_tax.get())
        #print(self.epoch.get())
        pass

    def test_mlp(self):
        """ Funcao que testa o algoritmo do Multilayer Perceptron. """
        #print(self.file_content_testing)
        pass



if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
