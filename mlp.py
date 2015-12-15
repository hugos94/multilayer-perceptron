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

    def __init__(self, window, architecture=[4, 4, 3]):
        ''' Inicializa a arquitetura da rede neural. '''

        self.architecture = architecture
        self.window = window

        self.neurons_in = []    # Neuronios da camada escondida
        self.neurons_out = []   # Neuronios da camada de saida

        # Cria a estrutura do MLP
        self.dot = Digraph(format='png')
        self.dot.body.extend(['rankdir=LR', 'size="8,5"'])

        # Alterando o tipo de no para cubo
        self.dot.attr('node', shape='box')

        # Cria as entradas dos neuronios
        for i in range(architecture[0]):
            self.dot.node("input"+str(i+1),"Input:"+ str(i) + "\nValue:" + str(0))

        # Alterando o tipo de no para circulo
        self.dot.attr('node', shape='circle')

        # Criando neuronios da camada escondida
        for i in range(architecture[1]):
            weights = []
            for j in range(architecture[0]):
                # Criando pesos com valores aleatorios
                weights.append(random.random())
            # Criando limiar de ativacao com valor aleatorio
            theta = random.random()
            self.neurons_in.append(Neuron(weights, theta))

            # Cria a estrutura do neuronio da camada escondida no grafo
            self.dot.node("hidden_layer"+str(i+1), "Neuron:"+str(i+1) + "\nSum:" + str(0))

        # Criando neuronios da camada de saida
        for i in range(architecture[2]):
            weights = []
            for j in range(architecture[1]):
                # Criando pesos com valores aleatorios
                weights.append(random.random())
            # Criando limiar de ativacao com valor aleatorio
            theta = random.random()
            self.neurons_out.append(Neuron(weights, theta))

            # Cria a estrutura do neuronio da camada de saida do grafo
            self.dot.node("out_layer"+str(i+1), "Neuron:"+str(i+1) + "\nSum:" + str(0))

        # Cria as ligacoes entre as entradas e a camada escondida
        for i in range(architecture[0]):
            for j in range(architecture[1]):
                self.dot.edge("input"+str(i+1),"hidden_layer"+str(j+1), label="\t\t" + str(round(self.neurons_in[j].weight[i],3)) + "\t\t")

        # Cria as ligacoes entre a camada escondida e a camada de saida
        for i in range(architecture[1]):
            for j in range(architecture[2]):
                self.dot.edge("hidden_layer"+str(i+1),"out_layer"+str(j+1), label="\t\t" + str(round(self.neurons_out[j].weight[i],3)) + "\t\t")

        # Cria a label que será mostrada no nó de saida
        label_out=""
        for i in range(architecture[2]):
            label_out += "Saida:" + str(i) + "\tValue:" + str(round(self.neurons_out[i].output,3)) + "\n"

        # Cria o no de saida n grafo
        self.dot.node("out",label_out)

        # Cria as ligacoes entre a camada de saida e a saida final
        for i in range(architecture[2]):
            self.dot.edge("out_layer"+(str(i+1)), "out", label="")

        self.dot.render(view=False, cleanup=False)

        self.label_input = StringVar()
        self.label_input.set(" Entrada: 0 ")
        tk.Label(self.window,textvariable=self.label_input).grid(column=0,row=0)
        tk.Button(self.window, text=" Executar entrada! ").grid(column=0,row=1)

        # Carrega a imagem
        imagem = ImageTk.PhotoImage(Image.open("Digraph.gv.png").convert("RGB"))

        # Cria uma label que ira conter a imagem da arvore de decisao
        label = tk.Label(self.window, image=imagem)
        label.image = imagem
        label.grid(column=1,row=2)


    def trainning(self, learning_tax, inputs, outputs):
        ''' Metodo que treina a rede neural de multiplas camadas. '''

        tam_inputs = len(inputs)
        for i in range(tam_inputs): # Iteracao em entradas
            self.label_input.set(" Entrada: " + str(i+1) + "  Restantes: " + str(tam_inputs-i-1))

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

        '''Criacao do grafo do MLP'''

        self.dot = Digraph(format='png')
        self.dot.body.extend(['rankdir=LR', 'size="8,5"'])

        # Atualiza as entradas do grafo
        for k in range(self.architecture[0]):
            self.dot.node("input"+str(k+1),"Input:"+ str(k) + "\nValue:" + str(inputs[i][k]))

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

        # Cria a label que será mostrada no nó de saida
        label_out=""
        for k in range(self.architecture[2]):
            label_out += "Saida:" + str(k) + "\tValue:" + str(round(self.neurons_out[k].output,3)) + "\n"
            self.dot.edge("out_layer"+(str(k+1)), "out", label="")

        # Cria o no de saida n grafo
        self.dot.node("out",label_out)

        '''Fim da criacao do grafo'''

        #""" TESTAR SE SAIDA EH IGUAL A ESPERADA """
        # Calcula o erro
        if not self.error():
            self.update_weights(outputs[i], learning_tax)

        self.dot.render(view=False, cleanup=True)

        # Carrega a imagem
        imagem = ImageTk.PhotoImage(Image.open("Digraph.gv.png").convert("RGB"))

        # Cria uma label que ira conter a imagem da arvore de decisao
        label = tk.Label(self.window, image=imagem)
        label.image = imagem
        label.grid(column=1,row=2)

    def test(self, inputs, outputs):
        for i in range(len(inputs)):
            print("Teste " + str(i))
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

                print(oup.output)

            print(outputs[i])


    def update_weights(self, outputs, n):
        ''' Metodo que atualiza os pesos da rede neural de multiplas camadas. '''
        out_error = []
        in_error = []

        # Calculando o erro da camada de saida
        for i in range(len(self.neurons_out)):
            out_error.append(math.pow((outputs[i] - self.neurons_out[i].output), 2.0)/2.0)

        # Calculando o erro da camada escondida
        for i in range(len(self.neurons_in)):
            error = 0

            for j in range(len(self.neurons_out)):
                error += out_error[j] * self.neurons_out[j].weight[i]

            in_error.append(error)

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


    def error(self):
        return False
