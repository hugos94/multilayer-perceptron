#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from graphviz import Digraph
from neuron import *
import random
import copy
import time

class MLP(object):


    def __init__(self, architecture=[4, 4, 3]):

        self.architecture = architecture

        # Cria a estrutura do MLP
        self.dot = Digraph(format='png')
        self.dot.body.extend(['rankdir=LR', 'size="8,5"'])

        self.neurons_in = []    # Neuronios da camada escondida
        self.neurons_out = []   # Neuronios da camada de saida

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
            self.dot.node("hidden_layer"+str(i+1), "Neuron:"+str(i+1) + "\Sum:" + str(0))


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

        # Cria o no de saida n grafo
        self.dot.node("out","out")

        # Cria as ligacoes entre as entradas e a camada escondida
        for i in range(architecture[0]):
            for j in range(architecture[1]):
                self.dot.edge("input"+str(i+1),"hidden_layer"+str(j+1), label="\t\t" + str(round(self.neurons_in[j].weight[i],3)) + "\t\t")

        # Cria as ligacoes entre a camada escondida e a camada de saida
        for i in range(architecture[1]):
            for j in range(architecture[2]):
                self.dot.edge("hidden_layer"+str(i+1),"out_layer"+str(j+1), label="\t\t" + str(round(self.neurons_out[j].weight[i],3)) + "\t\t")

        # Cria as ligacoes entre a camada de saida e a saida final
        for i in range(architecture[2]):
            self.dot.edge("out_layer"+(str(i+1)), "out", label="\t\t" +str(self.neurons_out[i].output) + "\t\t")

        # Renderiza a arvore de decisao
        self.dot.render(view=True, cleanup=True)

        time.sleep(1)


    def trainning(self, epoch, learning_tax, inputs, outputs):
        # for oup in self.neurons_out:
        #     print (oup.output)
        for j in range(epoch+1):
            for i in range(len(inputs)):

                self.dot = Digraph(format='png')
                self.dot.body.extend(['rankdir=LR', 'size="8,5"'])

                inputs_out = []

                # Atualiza as entradas do grafo
                for k in range(self.architecture[0]):
                    self.dot.node("input"+str(k+1),"Input:"+ str(k) + "\nValue:" + str(inputs[i][k]))

                for inp in self.neurons_in:
                    # Modifica os valores de entrada para os neuronios da camada escondida
                    inp.input = copy.deepcopy(inputs[i])

                    # Recalcula a saida do neuronio com as novas entradas
                    inp.recalculate_output()

                    # Valores de entrada para os neuronios da camada de saida
                    inputs_out.append(inp.output)

                # Atualiza as entradas da camada escondida
                for k in range(self.architecture[1]):
                    self.dot.node("hidden_layer"+str(k+1),"Neuron:"+ str(k) + "\nSum:" + str(round(self.neurons_in[k].sum_inputs(),3)))

                for oup in self.neurons_out:
                    # Modifica os valores de entrada para os neuronios da camada de saida
                    oup.input = copy.deepcopy(inputs_out)

                    # Recalcula a saida do neuronio com as novas entradas
                    oup.recalculate_output()

                # Atualiza as entradas da camada de saida
                for k in range(self.architecture[2]):
                    self.dot.node("out_layer"+str(k+1),"Neuron:"+ str(k) + "\nSum:" + str(round(self.neurons_out[k].sum_inputs(),3)))

                # Cria as ligacoes entre as entradas e a camada escondida
                for l in range(self.architecture[0]):
                    for m in range(self.architecture[1]):
                        self.dot.edge("input"+str(l+1),"hidden_layer"+str(m+1), label="\t\t" + str(round(self.neurons_in[m].weight[l],3)) + "\t\t")

                # Cria as ligacoes entre a camada escondida e a camada de saida
                for l in range(self.architecture[1]):
                    for m in range(self.architecture[2]):
                        self.dot.edge("hidden_layer"+str(l+1),"out_layer"+str(m+1), label="\t\t" + str(round(self.neurons_out[m].weight[l],3)) + "\t\t")

                # Cria as ligacoes entre a camada de saida e a saida final
                for k in range(self.architecture[2]):
                    self.dot.edge("out_layer"+(str(k+1)), "out", label="\t\t" +str(round(self.neurons_out[k].output,3)) + "\t\t")

                # Renderiza a arvore de decisao
                self.dot.render(view=True, cleanup=True)

                time.sleep(1)

                #""" TESTAR SE SAIDA EH IGUAL A ESPERADA """
                # Calcula o erro
                if not self.error():
                    self.update_weights(outputs[i], learning_tax)

        #self.dot.render(view=True, cleanup=True)


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
        out_error = []
        in_error = []

        # Calculando o erro da camada de saida
        for i in range(len(self.neurons_out)):
            out_error.append(math.fabs(outputs[i] - self.neurons_out[i].output))

        # Calculando o erro da camada escondida
        for i in range(len(self.neurons_in)):
            error = 0

            for j in range(len(self.neurons_out)):
                error += out_error[j] * self.neurons_out[j].weight[i]

            in_error.append(error)

        # Calculando os novos pesos e o novo limiar de ativacao da camada escondida
        for i,inp in enumerate(self.neurons_in):
            # self.neurons_in[i].theta = self.neurons_in[i].theta + (n * in_error[i] * inp.calculate_derived_sigmoid() * inp.input[j])
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
