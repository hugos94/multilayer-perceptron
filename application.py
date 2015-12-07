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

        # Cria o botao para criar a arquitetura do MLP
        tk.Button(self, text="Arquitetura", command=self.set_architecture).grid(column=3,row=0)

        # Cria o botao para treinar o MLP
        tk.Button(self, text='Treinar Rede Neural...', command=self.verify_trainning_mlp).grid(column = 4, row = 0)

        # Cria o botao para abrir o arquivo de teste
        tk.Button(self, text='Abrir Arquivo de Teste...', command=self.open_file_test).grid(column = 5, row = 0)

        # Cria o botao para testar o MLP
        tk.Button(self, text='Testar Rede Neural...', command=self.test_mlp).grid(column = 6, row = 0)


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
            tk.Label(self, text=name).grid(column=5,row=1)


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
        else:
            tk.messagebox.showwarning("Quantidade de epocas nao informada", "Informe a quantidade de epocas para continuar!")


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
        """ Metodo que verifica se a taxa de aprendizagem foi inserida. """

        if self.learning_tax.get():
            tk.Label(self,text=self.learning_tax.get()).grid(column=2,row=1)
            self.window.destroy()
        else:
            tk.messagebox.showwarning("Taxa de Aprendizagem nao informada", "Informe taxa de aprendizagem para continuar!")


    def set_architecture(self):
        """ Cria uma nova janela pra inserir a arquitetura do MLP. """

        self.window = tk.Toplevel(self)
        self.window.title("Arquitetura do MLP!")
        self.window.grid

        tk.Label(self.window, text="Informe a quantidade de entradas:").grid(column=0,row=0)
        self.in_layer = StringVar()
        tk.Entry(self.window,width=20, textvariable=self.in_layer).grid(column=1,row=0)

        tk.Label(self.window, text="Informe a quantidade de neuronios na camada escondida:").grid(column=0,row=1)
        self.hidden_layer = StringVar()
        tk.Entry(self.window,width=20, textvariable=self.hidden_layer).grid(column=1,row=1)

        tk.Label(self.window, text="Informe a quantidade de neuronios na camada de saida:").grid(column=0,row=2)
        self.out_layer = StringVar()
        tk.Entry(self.window,width=20, textvariable=self.out_layer).grid(column=1,row=2)

        tk.Button(self.window,text="Salvar!", command=self.verify_architecture).grid(column=2,row=1)


    def verify_architecture(self):
        """ Metodo que verifica se a arquitetura foi inserida """

        if self.in_layer.get():
            if self.hidden_layer.get():
                if self.out_layer.get():
                    self.architecture_type = self.in_layer.get() + "-" + self.hidden_layer.get() + "-" + self.out_layer.get()
                    tk.Label(self,text=self.architecture_type).grid(column=3,row=1)
                    self.window.destroy()
                else:
                    tk.messagebox.showwarning("Numero de neuronios na camada de saida nao informado", "Informe o numero de neuronios na camada de saida para continuar!")
            else:
                tk.messagebox.showwarning("Numero de neuronios na camada escondida nao informado", "Informe o numero de neuronios na camada escondida para continuar!")
        else:
            tk.messagebox.showwarning("Numero de entradas nao informado", "Informe o numero de entradas para continuar!")


    def verify_trainning_mlp(self):
        """ Verifica se todos os atributos para treinar o MLP foram inseridos. """
        try:
            if self.file_content_trainning:
                try:
                    if self.epoch.get():
                        try:
                            if self.learning_tax.get():
                                self.trainning_mlp()
                        except AttributeError:
                            tk.messagebox.showwarning("Taxa de Aprendizagem nao informada", "Informe a taxa de aprendizagem para continuar!")
                except AttributeError:
                    tk.messagebox.showwarning("Quantidade de epocas nao informado", "Informe a quantidade de epocas para continuar!")
        except AttributeError:
            tk.messagebox.showwarning("Arquivo de treinamento nao informado", "Informe o arquivo de treinamento para continuar!")

    def trainning_mlp(self):
        """ Funcao que treina o algoritmo do Multilayer Perceptron. """

        fm = FileManager()

    	# Le os dados de entrada a partir de um arquivo csv
        file_content = fm.read_csv(self.file_content_trainning)

        # Remove a lista de atributos do arquivo
        attributes = Matrix.extract_attributes(file_content)

        # Seleciona quantidade de linhas a serem utilizadas
        file_content = Matrix.get_rows_matrix(file_content, 0, 5)

        # Devolve colunas com as entradas
        inputs = Matrix.remove_columns_2(file_content, [4,5,6])

        # Devolve colunas com as saidas esperadas
        outputs = Matrix.remove_columns_2(file_content, [0,1,2,3])

        # Converte elementos das matrizes em float
        inputs = Matrix.to_float(inputs)
        outputs = Matrix.to_float(outputs)

        # Imprime matriz a ser utilizada
        print("Matriz de entrada: ", end='')
        Matrix.print_matrix(inputs)

        e = epoch

        while e > 0:   #""" (DO WHILE) OU ESTA NA FAIXA DE ERRO ACEITO """
            e -= 1

            for i,line in enumerate(inputs):
                if e == (epoch - 1) and i == 0:
                    # Utiliza primeira linha das entradas como inicializacao do MLP
                    mlp = MLP(inputs[i])
                else:
                    # Executa a rede MLP
                    mlp.execute(line)

                #""" TESTAR SE SAIDA EH IGUAL A ESPERADA """
                # Calcula o erro
                if not mlp.error():
                    mlp.update_weights(outputs[i], n)

                if e == 0:
                    print("---: NEURONIO", i, ":---")
                    for neu in mlp.neurons_out:
                        print(neu.output)
                    print()

                    # for i,inp in enumerate(mlp.neurons_in):
                    #     print("Neuronio da escondida (", i, ")\n", inp)
                    # for o,out in enumerate(mlp.neurons_out):
                    #     print("Neuronios de saida (", o, ")\n", out)



    def test_mlp(self):
        """ Funcao que testa o algoritmo do Multilayer Perceptron. """

        #print(self.file_content_testing)
        pass


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
