#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import *
import random
import copy

class MLP(object):


    def __init__(self, inputs, architecture=[4, 4, 3]):
        self.neurons_in = []    # Neuronios da camada escondida
        self.neurons_out = []   # Neuronios da camada de saida

        inputs_out = []

        # Criando neuronios da camada escondida
        for i in range(architecture[1]):
            weights_in = []
            for j in range(architecture[0]):
                # Criando pesos com valores aleatorios
                weights_in.append(random.random())

            # Criando limiar  de ativacao com valor aleatorio
            theta_in = random.random()
            self.neurons_in.append(Neuron(inputs, weights_in, theta_in))

            # Entradas para os neuronios da camada de saida
            inputs_out.append(self.neurons_in[i].output)

        # Criando neuronios da camada de saida
        for i in range(architecture[2]):
            weights_out = []
            for j in range(architecture[1]):
                # Criando pesos com valores aleatorios
                weights_out.append(random.random())

            # Criando limiar de ativacao com valor aleatorio
            theta_out = random.random()
            self.neurons_out.append(Neuron(inputs_out, weights_out, theta_out))


    def execute(self, inputs):
        inputs_out = []

        for inp in self.neurons_in:
            # Modifica os valores de entrada para os neuronios da camada escondida
            inp.input = copy.deepcopy(inputs)

            # Recalcula a saida do neuronio com as novas entradas
            inp.recalculate_output()

            # Valores de entrada para os neuronios da camada de saida
            inputs_out.append(inp.output)

        for i in range(len(self.neurons_out)):
            # Modifica os valores de entrada para os neuronios da camada de saida
            self.neurons_out[i].input = copy.deepcopy(inputs_out)

            # Recalcula a saida do neuronio com as novas entradas
            self.neurons_out[i].recalculate_output()



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
