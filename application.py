#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import StringVar
from tkinter import messagebox

from file_manager import *
import copy
from mlp import *
from matrix import *

from PIL import Image, ImageTk

class Application(tk.Frame):
    """docstring for Application"""

    global counter
    counter = 0


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
        tk.Button(self, text='Arquivo de Treinamento', command=self.open_file_trainning).grid(column = 0, row = 0)

        # Cria o botao para inserir a quantidade de epocas do treinamento
        tk.Button(self, text="Quantidade de Épocas", command=self.get_epoch).grid(column=1,row=0)

        self.epoch = StringVar()
        self.epoch.set("2500");
        tk.Label(self,textvariable=self.epoch).grid(column=1,row=1)

        # Cria o botao para inserir a taxa de aprendizagem do treinamento
        tk.Button(self, text="Taxa de Aprendizagem", command=self.get_learning_tax).grid(column=2,row=0)

        self.learning_tax = StringVar()
        self.learning_tax.set("0.5")
        tk.Label(self,textvariable=self.learning_tax).grid(column=2,row=1)

        # Cria o botao para criar a arquitetura do MLP dinamica
        #tk.Button(self, text="Arquitetura", command=self.set_architecture).grid(column=3,row=0)

        # Cria o botao para inserir a precisa do treinamento
        tk.Button(self, text="Precisão", command=self.get_precision_tax).grid(column=3,row=0)

        self.precision_tax = StringVar()
        self.precision_tax.set("0.05")
        tk.Label(self,textvariable=self.precision_tax).grid(column=3,row=1)

        # Cria o botao apenas como label para a arquitetura estatica do MLP
        tk.Button(self, text="Arquitetura", state=tk.DISABLED).grid(column=4,row=0)

        # Label que mostra a arquitetura utilizada ( estático )
        tk.Label(self,text="4-4-3").grid(column=4,row=1)

        # Cria o botao para treinar o MLP
        tk.Button(self, text='Treinar Rede', command=self.verify_trainning_mlp).grid(column = 5, row = 0)

        # Cria o botao para abrir o arquivo de teste
        tk.Button(self, text='Arquivo de Teste', command=self.open_file_test).grid(column = 6, row = 0)

        # Cria o botao para testar o MLP
        tk.Button(self, text='Testar Rede', command=self.test_mlp).grid(column = 7, row = 0)

         # Inicializa a combobox como nulo
        self.box = None


    def open_file_trainning(self):
        """Abre um File Dialog que retorna o nome do arquivo de treinamento"""

        global counter

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

            # Chama o metodo para criar a combobox
            self.create_combo_box()

            # Variavel que controla a execucao
            counter = 0

            tk.Label(self, text="").grid(column=5,row=1)


    def create_combo_box(self):
        """Cria a combobox para a escolha da quantidade de linhas do arquivo de treinamento a serem utilizadas."""
        value = StringVar()
        self.box = ttk.Combobox(self, textvariable=value, state='readonly')
        self.box['values'] = list(range(1,len(self.file_content_trainning)))
        if self.box['values']:
            self.box.current(0)
        self.box.grid(column = 0, row = 2)


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
            tk.Label(self, text=name).grid(column=7,row=1)

            self.test_flag = 0


    def get_epoch(self):
        """ Cria uma nova janela pra receber a quantidade de epocas para o treinamento. """

        self.window = tk.Toplevel(self)
        self.window.title("Quantidade de Epocas!")
        self.window.grid

        tk.Label(self.window, text="Informe a quantidade de épocas:").grid(column=0,row=0)
        tk.Entry(self.window,width=20, textvariable=self.epoch).grid(column=1,row=0)
        tk.Button(self.window,text="Salvar!", command=self.verify_entry_epoch).grid(column=2,row=0)


    def verify_entry_epoch(self):
        """ Metodo que verifica se a quantidade de epocas foi inserida. """

        if self.epoch.get():
            self.window.destroy()
        else:
            tk.messagebox.showwarning("Quantidade de epocas nao informada", "Informe a quantidade de epocas para continuar!")


    def get_learning_tax(self):
        """ Cria uma nova janela pra receber a taxa de aprendizagem para o treinamento. """

        self.window = tk.Toplevel(self)
        self.window.title("Taxa de Aprendizagem!")
        self.window.grid

        tk.Label(self.window, text="Informe a taxa de aprendizagem:").grid(column=0,row=0)
        tk.Entry(self.window,width=20, textvariable=self.learning_tax).grid(column=1,row=0)
        tk.Button(self.window,text="Salvar!", command=self.verify_entry_learning_tax).grid(column=2,row=0)


    def verify_entry_learning_tax(self):
        """ Metodo que verifica se a taxa de aprendizagem foi inserida. """

        if self.learning_tax.get():
            self.window.destroy()
        else:
            tk.messagebox.showwarning("Taxa de Aprendizagem nao informada", "Informe taxa de aprendizagem para continuar!")


    def get_precision_tax(self):
        """ Cria uma nova janela pra receber a taxa de precisao para o treinamento. """

        self.window = tk.Toplevel(self)
        self.window.title("Taxa de Precisão!")
        self.window.grid()

        tk.Label(self.window, text="Informe a taxa de precisão:").grid(column=0,row=0)
        tk.Entry(self.window,width=20, textvariable=self.precision_tax).grid(column=1,row=0)
        tk.Button(self.window,text="Salvar!", command=self.verify_entry_precision_tax).grid(column=2,row=0)


    def verify_entry_precision_tax(self):
        """ Metodo que verifica se a taxa de precision foi inserida. """

        if self.precision_tax.get():
            self.window.destroy()
        else:
            tk.messagebox.showwarning("Taxa de precisao nao informada", "Informe taxa de precisao para continuar!")


    def verify_trainning_mlp(self):
        """ Verifica se todos os atributos para treinar o MLP foram inseridos. """
        try:
            if self.file_content_trainning:
                try:
                    if self.epoch.get():
                        try:
                            if self.learning_tax.get():
                                try:
                                    if self.epoch.get():
                                        self.trainning_mlp()
                                except AttributeError:
                                    tk.messagebox.showwarning("Taxa de Precisao nao informada", "Informe a taxa de precisao para continuar!")
                        except AttributeError:
                            tk.messagebox.showwarning("Taxa de Aprendizagem nao informada", "Informe a taxa de aprendizagem para continuar!")
                except AttributeError:
                    tk.messagebox.showwarning("Quantidade de epocas nao informado", "Informe a quantidade de epocas para continuar!")
        except AttributeError:
            tk.messagebox.showwarning("Arquivo de treinamento nao informado", "Informe o arquivo de treinamento para continuar!")


    def trainning_mlp(self):
        """ Funcao que treina o algoritmo do Multilayer Perceptron. """

        global counter

        if self.box.get():
            if counter == 0:

                # Colocar o contador em 1 para evitar que o programa execute novamente
                counter = 1

                # Remove a lista de atributos do arquivo
                attributes = Matrix.extract_attributes(self.file_content_trainning)

                # Seleciona quantidade de linhas a serem utilizadas
                self.file_content_trainning = Matrix.get_rows_matrix(self.file_content_trainning, 0, int(self.box.get())-1)

                # Devolve colunas com as entradas
                self.inputs = Matrix.remove_columns_2(self.file_content_trainning, [4,5,6])

                # Devolve colunas com as saidas esperadas
                self.outputs = Matrix.remove_columns_2(self.file_content_trainning, [0,1,2,3])

                # Converte elementos das matrizes em float
                self.inputs = Matrix.to_float(self.inputs)
                self.outputs = Matrix.to_float(self.outputs)

                # Imprime matriz a ser utilizada
                print("Matriz de entrada: ", end='')
                Matrix.print_matrix(self.inputs)

                # Cria a sub janela que ira controlar a execucao do treinamento
                window_trainning = tk.Toplevel(self)
                window_trainning.title("Multilayer Perceptron Trainning")
                window_trainning.grid()

                self.label_epoch = StringVar()
                self.label_epoch.set(" Epoca: 0  Restantes: " + str(self.epoch.get()))
                tk.Label(window_trainning,textvariable=self.label_epoch).grid(column=0,row=0)
                self.button_epoch = tk.Button(window_trainning, text=" Executar por epoca? ", command=self.trainning_by_epoch)
                self.button_epoch.grid(column=0,row=1)

                self.button_all = tk.Button(window_trainning, text=" Executar completamente? ", command=self.trainning_all_epoch)
                self.button_all.grid(column=0,row=2)

                self.mlp = MLP(window_trainning,float(self.precision_tax.get()))

                self.epoch = int(self.epoch.get())
                self.epoch_counter = 0
                self.execute_flag = 0

            else:
                # Mensagem de erro gerada quando tentamos executar o algoritmo novamente
                tk.messagebox.showwarning("Carregar o arquivo de treinamento novamente", "Para executar, necessario carregar o arquivo de treinamento novamente.")
        else:
            # Mensagem de erro gerada quando tentamos executar o algoritmo sem escolher o atributo classe
            tk.messagebox.showwarning("Quantidade de elementos de treinamento nao escolhido", "Escolha a quantidade para continuar!")


    def trainning_by_epoch(self):
        if (self.epoch_counter < self.epoch):
            self.epoch_counter += 1
            self.label_epoch.set(" Epoca: " + str(self.epoch_counter) + "  Restantes: " + str(self.epoch-self.epoch_counter))
            self.mlp.trainning(float(self.learning_tax.get()), self.inputs, self.outputs, 0, 0)
        else:
            self.execute_flag = 1
            self.button_epoch.config(state=tk.DISABLED)
            self.button_all.config(state=tk.DISABLED)
            tk.Label(self, text="Rede Treinada!").grid(column=4,row=1)


    def trainning_all_epoch(self):
        flag = 0
        while (self.epoch_counter < self.epoch):
            self.epoch_counter += 1
            self.label_epoch.set(" Epoca: " + str(self.epoch_counter) + "  Restantes: " + str(self.epoch-self.epoch_counter))
            if(self.epoch_counter == self.epoch):
                flag = 1
            self.mlp.trainning(float(self.learning_tax.get()), self.inputs, self.outputs, 1, flag)
        self.execute_flag = 1
        self.button_epoch.config(state=tk.DISABLED)
        self.button_all.config(state=tk.DISABLED)
        tk.Label(self, text="Rede Treinada!").grid(column=4,row=1)


    def test_mlp(self):
        """ Funcao que testa o algoritmo do Multilayer Perceptron. """
        try:
            if(self.execute_flag == 1):
                try:
                    if self.file_content_testing:
                        try:
                            if (self.test_flag == 0):
                                self.test_flag = 1
                                self.window_test = tk.Toplevel(self)
                                self.window_test.title("Teste do Multilayer Perceptron!")
                                self.window_test.grid
                            else:
                                tk.messagebox.showwarning("Arquivo de teste ja utilizado!", "Informe o arquivo de teste novamente para continuar!")
                        except AttributeError:
                            tk.messagebox.showwarning("Arquivo de teste ja utilizado!", "Informe o arquivo de teste novamente para continuar!")
                except AttributeError:
                    tk.messagebox.showwarning("Arquivo de teste nao informado", "Informe o arquivo de teste para continuar!")
            else:
                tk.messagebox.showwarning("A rede Neural nao foi treinada!", "Treine a Rede Neural para continuar.")
        except AttributeError:
            tk.messagebox.showwarning("A rede Neural nao foi treinada!", "Treine a Rede Neural para continuar.")


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

'''
# Metodos que adicionam a funcionalidade de escolher a arquitetura dinamicamente

# def set_architecture(self):
#     """ Cria uma nova janela pra inserir a arquitetura do MLP. """
#
#     self.window = tk.Toplevel(self)
#     self.window.title("Arquitetura do MLP!")
#     self.window.grid
#
#     tk.Label(self.window, text="Informe a quantidade de entradas:").grid(column=0,row=0)
#     self.in_layer = StringVar()
#     tk.Entry(self.window,width=20, textvariable=self.in_layer).grid(column=1,row=0)
#
#     tk.Label(self.window, text="Informe a quantidade de neuronios na camada escondida:").grid(column=0,row=1)
#     self.hidden_layer = StringVar()
#     tk.Entry(self.window,width=20, textvariable=self.hidden_layer).grid(column=1,row=1)
#
#     tk.Label(self.window, text="Informe a quantidade de neuronios na camada de saida:").grid(column=0,row=2)
#     self.out_layer = StringVar()
#     tk.Entry(self.window,width=20, textvariable=self.out_layer).grid(column=1,row=2)
#
#     tk.Button(self.window,text="Salvar!", command=self.verify_architecture).grid(column=2,row=1)

# def verify_architecture(self):
#     """ Metodo que verifica se a arquitetura foi inserida """
#
#     if self.in_layer.get():
#         if self.hidden_layer.get():
#             if self.out_layer.get():
#                 self.architecture_type = self.in_layer.get() + "-" + self.hidden_layer.get() + "-" + self.out_layer.get()
#                 tk.Label(self,text=self.architecture_type).grid(column=3,row=1)
#                 self.window.destroy()
#             else:
#                 tk.messagebox.showwarning("Numero de neuronios na camada de saida nao informado", "Informe o numero de neuronios na camada de saida para continuar!")
#         else:
#             tk.messagebox.showwarning("Numero de neuronios na camada escondida nao informado", "Informe o numero de neuronios na camada escondida para continuar!")
#     else:
#         tk.messagebox.showwarning("Numero de entradas nao informado", "Informe o numero de entradas para continuar!")
#
'''
