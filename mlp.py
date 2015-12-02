#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import *

class MLP(object):
    qntd_out = 3        # Quantidade de saidas
    neurons_in = []        # Neuronios da camada escondida
    neurons_out = []    # Neuronios da camada de saida

    def initialize(self, inputs, weights, mi):
        # Imprime linha a ser utilizada no momento
        # print("\nEntradas utilizadas: ", inputs)


        inputs_out = []
        for i in range(len(inputs)):
            # Criando neuronios da camada escondida
            self.neurons_in.append(Neuron(inputs, weights, mi))
            # Preparando entradas dos neuronios da camada de saida
            inputs_out.append(self.neurons_in[i].output)

        for i in range(self.qntd_out):
            # Criando neuronios da camada de saida
            self.neurons_out.append(Neuron(inputs_out, weights, mi))
            # print("neurons_out.output[",i,"] = ", self.neurons_out[i].output)

    def training(self, outputs, n):
        """
            Treinamento do neuronios da camada de saida
        """
        for i,n_out in enumerate(self.neurons_out):

            # Atualiza o potencial de ativacao dos neuronios de saida
            self.neurons_out[i].mi = n_out.mi + (n * (n_out.output - outputs[i]) * -1)

            for j,w_out in enumerate(n_out.weight):
                # Atualiza os pesos dos neuronios de saida
                self.neurons_out[i].weight[j] = w_out + (n * (n_out.output - outputs[i]) * n_out.input[j])

            # Recalcula o valor de saida dos neuronios de saida
            self.neurons_out[i].recalculate_output()

        """
            Treinamento do neuronios da camada escondida
        """
        for i,n_in in enumerate(self.neurons_in):

            # Atualiza o potencial de ativacao dos neuronios de saida
            self.neurons_in[i].mi = n_in.mi + (n * (n_in.output - outputs[i]) * -1)

            for j,w_in in enumerate(n_in.weight):
                # Atualiza os pesos dos neuronios de saida
                self.neurons_in[i].weight[j] = w_in + (n * (n_in.output - outputs[i]) * n_in.input[j])

            # Recalcula o valor de saida dos neuronios de saida
            self.neurons_in[i].recalculate_output()
