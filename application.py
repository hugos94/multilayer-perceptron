#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import StringVar
from tkinter import messagebox

from file_manager import *
import copy
import csv
from mlp import *
from matrix import *

from PIL import Image, ImageTk

class Application(tk.Frame):
    """docstring for Application"""

    global trainning_counter
    trainning_counter = 0


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

        # Cria a label que informa a quantidade de epocas e vem com 2500 epocas setada como padrao
        self.epoch = StringVar()
        self.epoch.set("2500");
        tk.Label(self,textvariable=self.epoch).grid(column=1,row=1)

        # Cria o botao para inserir a taxa de aprendizagem do treinamento
        tk.Button(self, text="Taxa de Aprendizagem", command=self.get_learning_tax).grid(column=2,row=0)

        # Cria a label que informa a taxa de aprendizado e vem com 0.5 setado como padrao
        self.learning_tax = StringVar()
        self.learning_tax.set("0.5")
        tk.Label(self,textvariable=self.learning_tax).grid(column=2,row=1)

        # Cria o botao para inserir a taxa de precisao do treinamento
        tk.Button(self, text="Precisão", command=self.get_precision_tax).grid(column=3,row=0)

        # Cria a label que informa a taxa de precisa do treinamento e vem setada com 0.0000001
        self.precision_tax = StringVar()
        self.precision_tax.set("0.0000001")
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
        tk.Button(self, text='Testar Rede', command=self.verify_teste_mlp).grid(column = 7, row = 0)

         # Inicializa a combobox como nulo
        self.box = None

        # Inicializa o arquivo de treinamento como nulo
        self.file_content_trainning = None

        # Inicializa o arquivo de teste como nulo
        self.file_content_testing = None

        # Inicializa a variavel que verifica se o treinamento ocorreu com zero
        self.execute_flag = 0

        # Inicializa a variavel que evita a execucao de um novo teste sem carregar um novo arquivo
        self.test_flag = 0


    def open_file_trainning(self):
        """Abre um File Dialog que retorna o nome do arquivo de treinamento"""

        # Variavel que evita a execucao de um novo treinamento sem carregar um novo arquivo
        global trainning_counter

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

            # Variavel que evita a execucao de um novo treinamento sem carregar um novo arquivo
            trainning_counter = 0

            # Limpa a label que informa que o treinamento terminou
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
            tk.Label(self, text=name).grid(column=6,row=1)

            # Variavel que evita a execucao de um novo teste sem carregar um novo arquivo
            self.test_flag = 0


    def get_epoch(self):
        """ Cria uma nova janela pra receber a quantidade de epocas para o treinamento. """

        self.window_epoch = tk.Toplevel(self)
        self.window_epoch.title("Quantidade de Epocas!")
        self.window_epoch.grid

        tk.Label(self.window_epoch, text="Informe a quantidade de épocas: ").grid(column=0,row=0)
        tk.Entry(self.window_epoch,width=20, textvariable=self.epoch).grid(column=1,row=0)
        tk.Button(self.window_epoch,text="Salvar!", command=self.verify_entry_epoch).grid(column=2,row=0)


    def verify_entry_epoch(self):
        """ Metodo que verifica se a quantidade de epocas foi inserida. """

        # Verifica se a epoca foi inserida
        if self.epoch.get():
            #Destroi a janela que recebe a epoca
            self.window_epoch.destroy()
        else:
            # Mensagem de erro informando que a epoca nao foi inserida
            tk.messagebox.showwarning("Quantidade de epocas nao informada", "Informe a quantidade de epocas para continuar!")


    def get_learning_tax(self):
        """ Cria uma nova janela pra receber a taxa de aprendizagem para o treinamento. """

        self.window_learning = tk.Toplevel(self)
        self.window_learning.title("Taxa de Aprendizado!")
        self.window_learning.grid

        tk.Label(self.window_learning, text="Informe a taxa de aprendizado: ").grid(column=0,row=0)
        tk.Entry(self.window_learning,width=20, textvariable=self.learning_tax).grid(column=1,row=0)
        tk.Button(self.window_learning,text="Salvar!", command=self.verify_entry_learning_tax).grid(column=2,row=0)


    def verify_entry_learning_tax(self):
        """ Metodo que verifica se a taxa de aprendizagem foi inserida. """

        # Verifica se a taxa de aprendizado foi inserida
        if self.learning_tax.get():
            # Destroi a janela que recebe a taxa de aprendizado
            self.window_learning.destroy()
        else:
            # Mensagem de erro informando que a taxa de aprendizado nao foi inserida
            tk.messagebox.showwarning("Taxa de Aprendizado nao informada", "Informe taxa de aprendizado para continuar!")


    def get_precision_tax(self):
        """ Cria uma nova janela pra receber a taxa de precisao para o treinamento. """

        self.window_precision = tk.Toplevel(self)
        self.window_precision.title("Taxa de Precisão!")
        self.window_precision.grid()

        tk.Label(self.window_precision, text="Informe a taxa de precisão:").grid(column=0,row=0)
        tk.Entry(self.window_precision,width=20, textvariable=self.precision_tax).grid(column=1,row=0)
        tk.Button(self.window_precision,text="Salvar!", command=self.verify_entry_precision_tax).grid(column=2,row=0)


    def verify_entry_precision_tax(self):
        """ Metodo que verifica se a taxa de precisao foi inserida. """

        # Verifica se a taxa de precisao foi inserida
        if self.precision_tax.get():
            # Destroi a janela que recebe a taxa de precisao
            self.window_precision.destroy()
        else:
            # Mensagem de erro que informa que a taxa de precisao nao foi informada
            tk.messagebox.showwarning("Taxa de precisao nao informada", "Informe taxa de precisao para continuar!")


    def verify_trainning_mlp(self):
        """ Metodo que verifica se todos os atributos para treinar o MLP foram inseridos. """

        global trainning_counter

        if trainning_counter == 0: # Verifica se a rede ja foi treinada
            if self.file_content_trainning: # Verifica se o arquivo de treinamento foi carregado
                if self.epoch.get(): # Verifica se a variavel de epoca foi informada
                    if self.learning_tax.get(): # Verifica se a taxa de aprendizado foi informada
                        if self.precision_tax.get(): # Verifica se a taxa de precisao foi informada
                            if self.box.get(): # Verifica se foram escolhidas as quantidades de elementos do treinamento
                                self.trainning_mlp() # Chama o metodo para treinar o mlp
                            else: # Mensagem de erro gerada quando a quantidade de elementos do treinamento nao foi informado
                                tk.messagebox.showwarning("Quantidade de elementos de treinamento nao escolhido", "Escolha a quantidade de elementos de treinamento para continuar!")
                        else: # Mensagem de erro gerada quando a taxa de precisao nao foi informada
                            tk.messagebox.showwarning("Taxa de Precisao nao informada", "Informe a taxa de precisao para continuar!")
                    else: # Mensagem de erro gerada quando a taxa de aprendizado nao foi informada
                        tk.messagebox.showwarning("Taxa de Aprendizado nao informada", "Informe a taxa de aprendizado para continuar!")
                else: # Mensagem de erro gerada quando a quantidade de epocas nao foi informada
                    tk.messagebox.showwarning("Quantidade de epocas nao informado", "Informe a quantidade de epocas para continuar!")
            else: # Mensagem de erro gerada quando o arquivo de treinamento nao foi informado
                tk.messagebox.showwarning("Arquivo de treinamento nao informado", "Informe o arquivo de treinamento para continuar!")
        else: # Mensagem de erro gerada quando tentamos executar o algoritmo novamente
            tk.messagebox.showwarning("Carregar o arquivo de treinamento novamente", "Para executar, necessario carregar o arquivo de treinamento novamente.")

    def trainning_mlp(self):
        """ Metodo que chama os metodos de treinamento por epoca do algoritmo do Multilayer Perceptron. """

        global trainning_counter

        # Colocar o contador em 1 para evitar que o programa execute novamente
        trainning_counter = 1

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

        # Cria a sub janela que ira controlar a execucao do treinamento
        window_trainning = tk.Toplevel(self)
        window_trainning.title("Multilayer Perceptron Trainning")
        window_trainning.grid()

        # Cria a label que mostrara a epoca atual e a quantidade de epocas restantes
        self.label_epoch = StringVar()
        self.label_epoch.set(" Epoca: 0  Restantes: " + str(self.epoch.get()))
        tk.Label(window_trainning,textvariable=self.label_epoch).grid(column=0,row=0)

        # Cria o botao que executa o treinamento por epoca
        self.button_epoch = tk.Button(window_trainning, text=" Executar por epoca? ", command=self.trainning_by_epoch)
        self.button_epoch.grid(column=0,row=1)

        # Cria o botao que executa o treinamento completo
        self.button_all = tk.Button(window_trainning, text=" Executar completamente? ", command=self.trainning_all_epoch)
        self.button_all.grid(column=0,row=2)

        # Instacia e inicializa a classe MLP
        self.mlp = MLP(window_trainning,float(self.precision_tax.get()))

        self.epoch = int(self.epoch.get())
        self.epoch_counter = 0
        self.execute_flag = 0


    def trainning_by_epoch(self):
        ''' Metodo que treina a rede neural por epocas ao click do botao'''

        if (not self.mlp.stop_trainning() and self.epoch_counter < self.epoch):
            self.epoch_counter += 1
            self.label_epoch.set(" Epoca: " + str(self.epoch_counter) + "  Restantes: " + str(self.epoch-self.epoch_counter))
            self.mlp.trainning(float(self.learning_tax.get()), self.inputs, self.outputs, 0, 0)
        else:
            # Finaliza execucao
            self.execute_flag = 1
            self.button_epoch.config(state=tk.DISABLED)
            self.button_all.config(state=tk.DISABLED)
            tk.Label(self, text="Rede Treinada!").grid(column=4,row=1)

            # Tempo de execucao de todo o treinamento
            print("Tempo de execucao =", self.mlp.time, "segundos")
            # A rotina de escrita recebe um objeto do tipo file
            out_csv = open('Output.csv', 'w')
            for i, value in enumerate(self.mlp.data_csv):
                # Escrevendo as tuplas no arquivo
                if i % 2 == 0:
                    out_csv.write(str(value) + ',')
                else:
                    out_csv.write(str(value) + '\n')

            out_csv.close()


    def trainning_all_epoch(self):
        ''' Metodo que treina a rede neural completamente ao click do botao'''

        flag = 0
        while (not self.mlp.stop_trainning() and self.epoch_counter < self.epoch):
            self.epoch_counter += 1
            self.label_epoch.set(" Epoca: " + str(self.epoch_counter) + "  Restantes: " + str(self.epoch-self.epoch_counter))
            if(self.epoch_counter == self.epoch):
                flag = 1
            self.mlp.trainning(float(self.learning_tax.get()), self.inputs, self.outputs, 1, flag)

        # Finaliza execucao
        self.execute_flag = 1
        self.button_epoch.config(state=tk.DISABLED)
        self.button_all.config(state=tk.DISABLED)
        tk.Label(self, text="Rede Treinada!").grid(column=4,row=1)

        # Tempo de execucao de todo o treinamento
        print("Tempo de execucao =", self.mlp.time, "segundos")
        # A rotina de escrita recebe um objeto do tipo file
        out_csv = open('Output.csv', 'w')
        for i, value in enumerate(self.mlp.data_csv):
            # Escrevendo as tuplas no arquivo
            if i % 2 == 0:
                out_csv.write(str(value) + ',')
            else:
                out_csv.write(str(value) + '\n')

        out_csv.close()


    def verify_teste_mlp(self):
        """ Funcao que testa o algoritmo do Multilayer Perceptron. """

        if(self.execute_flag == 1): # Verifica se a rede neural ja foi treinada
            if self.file_content_testing: # Verifica se o arquivo de teste foi carregado
                if (self.test_flag == 0): # Verifica se o arquivo que foi carregado já foi testado
                    self.test_mlp() # Chama o algoritmo de teste da rede neural
                else:
                    tk.messagebox.showwarning("Arquivo de teste ja utilizado!", "Informe o arquivo de teste novamente para continuar!")
            else:
                tk.messagebox.showwarning("Arquivo de teste nao informado", "Informe o arquivo de teste para continuar!")
        else:
            tk.messagebox.showwarning("A rede Neural nao foi treinada!", "Treine a Rede Neural para continuar.")


    def test_mlp(self):
        ''' Metodo que extrai os atributos necessarios para testar o MLP '''

        # Incrementa a variavel que impede do algoritmo ser testado com um arquivo sem ser carregado novamente
        self.test_flag = 1

        # Remove a lista de atributos do arquivo
        attributes = Matrix.extract_attributes(self.file_content_testing)

        # Devolve colunas com as entradas
        self.inputs_test = Matrix.remove_columns_2(self.file_content_testing, [4,5,6])

        # Devolve colunas com as saidas esperadas
        self.outputs_test = Matrix.remove_columns_2(self.file_content_testing, [0,1,2,3])

        # Converte elementos das matrizes em float
        self.inputs_test = Matrix.to_float(self.inputs_test)
        self.outputs_test = Matrix.to_float(self.outputs_test)

        self.counter_test = 0

        # Cria a janela que testa o MLP
        self.window_test = tk.Toplevel(self)
        self.window_test.title("Teste do Multilayer Perceptron!")
        self.window_test.grid

        self.label_input_test = StringVar()
        self.label_input_test.set(" Entrada: "+ str(self.counter_test) + "  Restantes: " + str(len(self.inputs_test)))
        tk.Label(self.window_test,textvariable=self.label_input_test).grid(column=0,row=0)

        self.button_execute_test = tk.Button(self.window_test, text="Executar uma entrada?", command=self.execute_test_mlp)
        self.button_execute_test.grid(column=0,row=1)

        self.mlp.create_graph(self.window_test, [0] * 4)


    def execute_test_mlp(self):
        ''' Metodo que executa os testes no MLP '''
        if(self.counter_test < len(self.inputs_test)):
            self.mlp.testing(self.window_test, self.inputs_test[self.counter_test], self.outputs_test[self.counter_test])
            self.counter_test+=1
            self.label_input_test.set(" Entrada: "+ str(self.counter_test) + "  Restantes: " + str(len(self.inputs_test)-self.counter_test))

        else:
            self.button_execute_test.config(state=tk.DISABLED)

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
