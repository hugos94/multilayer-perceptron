#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import StringVar
from tkinter import messagebox

from graphviz import Digraph

from neuron import *
import random
import copy
import time

from PIL import Image, ImageTk

class MLP(object):
    ''' MLP e a classe responsavel por treinar e testar um Multilayer Perceptron. '''

    def __init__(self, window, precision, architecture=[4, 4, 3]):
        ''' Inicializa a arquitetura da rede neural. '''

        self.architecture = architecture
        self.window = window

        self.mean_error = 0.0       # Erro quadratico medio
        self.precision = precision  # Taxa de erro aceitavel

        # Dados que serao salvos no CSV de saida
        self.data_csv = ('Epoca', 'Erro medio quadratico')
        # Valor da epocas a serem salvas no arquivo
        self.epoch = 1

        self.time = 0               # Tempo de execucao do treinamento
        self.stop_flag = False      # Flag que informa se o erro ja e aceitavel

        self.neurons_in = []        # Neuronios da camada escondida
        self.neurons_out = []       # Neuronios da camada de saida

        # Criando neuronios da camada escondida
        for i in range(architecture[1]):
            weights = []
            for j in range(architecture[0]):
                # Criando pesos com valores aleatorios
                weights.append(random.random())
            # Criando limiar de ativacao com valor aleatorio
            theta = random.random()
            self.neurons_in.append(Neuron(weights, theta))

        # Criando neuronios da camada de saida
        for i in range(architecture[2]):
            weights = []
            for j in range(architecture[1]):
                # Criando pesos com valores aleatorios
                weights.append(random.random())
            # Criando limiar de ativacao com valor aleatorio
            theta = random.random()
            self.neurons_out.append(Neuron(weights, theta))

        self.create_graph(self.window, [0] * architecture[0])


    def trainning(self, learning_tax, inputs, outputs, by_epoch, flag):
        ''' Metodo que treina a rede neural de multiplas camadas. '''

        # Retorna o tempo inicial do treinamento dessa epoca
        start_time = time.time()

        # Atualiza erro anterior e zera erro medio
        previous_error = self.mean_error
        self.mean_error = 0.0

        tam_inputs = len(inputs)
        for i in range(tam_inputs): # Iteracao em entradas

            inputs_out = []
            for inp in self.neurons_in:
                # Modifica os valores de entrada para os neuronios da camada escondida
                inp.input = copy.deepcopy(inputs[i])

                # Recalcula a saida do neuronio com as novas entradas
                inp.recalculate_output()

                # Valores de entrada para os neuronios da camada de saida
                inputs_out.append(inp.output)

            for oup in self.neurons_out:
                # Modifica os valores de entrada para os neuronios da camada de saida
                oup.input = copy.deepcopy(inputs_out)

                # Recalcula a saida do neuronio com as novas entradas
                oup.recalculate_output()

            self.update_weights(outputs[i], learning_tax)

        # Atualiza erro atual e zera erro medio
        self.mean_error = self.mean_error / tam_inputs
        current_error = self.mean_error

        # Concatenacao dos dados a serem salvos
        self.data_csv += (self.epoch, self.mean_error)
        # Incremento do valor da epoca
        self.epoch += 1

        if math.fabs(current_error - previous_error) <= self.precision:
            self.stop_flag = True

        # Calcula o tempo de execucao do treinamento dessa epoca e somo ao tempo total
        self.time += time.time() - start_time

        if (by_epoch == 0 or flag == 1 or self.stop_flag == True):
            self.create_graph(self.window, inputs[i], outputs[i])


    def create_graph(self, window, inputs, outputs=[0, 0, 0]):
        '''Criacao do grafo do MLP'''

        self.dot = Digraph(format='png')
        self.dot.body.extend(['rankdir=LR', 'size="8,5"'])

        # Alterando o tipo de no para cubo
        self.dot.attr('node', shape='box')
        # Atualiza as entradas do grafo
        for k in range(self.architecture[0]):
            self.dot.node("input"+str(k+1),"Input:"+ str(k) + "\nValue:" + str(inputs[k]))

        # Alterando o tipo de no para cubo
        self.dot.attr('node', shape='circle')
        # Atualiza as entradas da camada escondida
        for k in range(self.architecture[1]):
            self.dot.node("hidden_layer"+str(k+1),"Neuron:"+ str(k) + "\nSum:" + str(round(self.neurons_in[k].sum_inputs(),3)))

        # Atualiza as entradas da camada de saida
        for k in range(self.architecture[2]):
            self.dot.node("out_layer"+str(k+1),"Neuron:"+ str(k) + "\nSum:" + str(round(self.neurons_out[k].sum_inputs(),3)))

        # Cria as ligacoes entre as entradas e a camada escondida
        for k in range(self.architecture[0]):
            for l in range(self.architecture[1]):
                self.dot.edge("input"+str(k+1),"hidden_layer"+str(l+1), label="\t\t" + str(round(self.neurons_in[l].weight[k],3)) + "\t\t")

        # Cria as ligacoes entre a camada escondida e a camada de saida
        for k in range(self.architecture[1]):
            for l in range(self.architecture[2]):
                self.dot.edge("hidden_layer"+str(k+1),"out_layer"+str(l+1), label="\t\t" + str(round(self.neurons_out[l].weight[k],3)) + "\t\t")

        # Alterando o tipo de no para cubo
        self.dot.attr('node', shape='box')

        # Cria a label que será mostrada no nó de saida
        label_out=""
        for k in range(self.architecture[2]):
            label_out += "Saida:" + str(k) + "  Value:" + str(round(self.neurons_out[k].output,3)) + "  Esperado: " + str(outputs[k]) + "\n"
            self.dot.edge("out_layer"+(str(k+1)), "out", label="")

        # Cria o no de saida n grafo
        self.dot.node("out",label_out)

        self.dot.render(view=False, cleanup=True)

        # Carrega a imagem
        imagem = ImageTk.PhotoImage(Image.open("Digraph.gv.png").convert("RGB"))

        # Cria uma label que ira conter a imagem da arvore de decisao
        label = tk.Label(window, image=imagem)
        label.image = imagem
        label.grid(column=0,row=3)

        '''Fim da criacao do grafo'''


    def testing(self, window_test, inputs, outputs):
        ''' Metodo que testa a rede neural. '''
        
        inputs_out = []

        for inp in self.neurons_in:
            # Modifica os valores de entrada para os neuronios da camada escondida
            inp.input = copy.deepcopy(inputs)

            # Recalcula a saida do neuronio com as novas entradas
            inp.recalculate_output()

            # Valores de entrada para os neuronios da camada de saida
            inputs_out.append(inp.output)

        for oup in self.neurons_out:
            # Modifica os valores de entrada para os neuronios da camada de saida
            oup.input = copy.deepcopy(inputs_out)

            # Recalcula a saida do neuronio com as novas entradas
            oup.recalculate_output()

        self.create_graph(window_test, inputs, outputs)

    def update_weights(self, outputs, n):
        ''' Metodo que atualiza os pesos da rede neural de multiplas camadas. '''

        out_error = []
        in_error = []

        # Variavel auxiliar para calcular o erro medio
        mean_error_aux = 0.0

        # Calculando o erro da camada de saida
        for i, out in enumerate(self.neurons_out):
            out_error.append((outputs[i] - out.output) * out.calculate_derived_sigmoid())

            mean_error_aux += math.pow((outputs[i] - out.output), 2.0)

        self.mean_error += mean_error_aux / 2.0

        # Calculando o erro da camada escondida
        for i, inp in enumerate(self.neurons_in):
            error = 0

            for j in range(len(self.neurons_out)):
                error += out_error[j] * self.neurons_out[j].weight[i]

            in_error.append(error * inp.calculate_derived_sigmoid())

        # Calculando os novos pesos e o novo limiar de ativacao da camada escondida
        for i,inp in enumerate(self.neurons_in):
            self.neurons_in[i].theta = self.neurons_in[i].theta + (n * in_error[i] * inp.calculate_derived_sigmoid() * inp.input[j])
            for j,wei in enumerate(inp.weight):
                # Calculando o novo peso
                self.neurons_in[i].weight[j] = wei + (n * in_error[i] * inp.calculate_derived_sigmoid() * inp.input[j])

        # Calculando os novos pesos e o novo limiar de ativacao da camada de saida
        for i,out in enumerate(self.neurons_out):
            for j,wei in enumerate(out.weight):
                # Calculando o novo peso
                self.neurons_out[i].weight[j] = wei + (n * out_error[i] * out.calculate_derived_sigmoid() * out.input[j])


    def stop_trainning(self):
        ''' Metodo que para o treinamento. '''
        return self.stop_flag
