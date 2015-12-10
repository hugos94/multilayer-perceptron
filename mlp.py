#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import *
import random
import copy

class MLP(object):


    def __init__(self, architecture=[4, 4, 3]):
        self.neurons_in = []    # Neuronios da camada escondida
        self.neurons_out = []   # Neuronios da camada de saida

        # Criando neuronios da camada escondida
        for i in range(architecture[1]):
            weights = []
            #print("Neuron In: " + str(i))
            for j in range(architecture[0]):
                # Criando pesos com valores aleatorios
                weights.append(random.random())

            #print(weights)
            # Criando limiar de ativacao com valor aleatorio
            theta = random.random()
            self.neurons_in.append(Neuron(weights, theta))

        # Criando neuronios da camada de saida
        for i in range(architecture[2]):
            #print("Neuron Out: " + str(i))

            weights = []
            for j in range(architecture[1]):
                # Criando pesos com valores aleatorios
                weights.append(random.random())
            #print(weights)
            # Criando limiar de ativacao com valor aleatorio
            theta = random.random()
            self.neurons_out.append(Neuron(weights, theta))


    def trainning(self, epoch, learning_tax, inputs, outputs):
        # for oup in self.neurons_out:
        #     print (oup.output)
        for j in range(epoch+1):
            for i in range(len(inputs)):
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

                #""" TESTAR SE SAIDA EH IGUAL A ESPERADA """
                # Calcula o erro
                if not self.error():
                    self.update_weights(outputs[i], learning_tax)

        #     print("Epoca " + str(j))
        #     for oup in self.neurons_out:
        #         print (oup.output)
        # print("Final:")
        # for oup in self.neurons_out:
        #     print (oup.output)

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
