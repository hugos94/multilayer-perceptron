#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import *

class MLP(object):
    qntd_out = 3   # Quantidade de saidas
    weight = 0.5   # Pesos sinaptico inicial

    def initialize(self, line):
        # Imprime linha a ser utilizada no momento
        print("\nEntradas utilizadas: ", line)

        # Neuronios da camada escondida
        neurons = []
        inputs_out = []
        for i in range(len(line)):
            # Criando neuronios da camada escondida
            neurons.append(Neuron(line, [self.weight, self.weight, self.weight, self.weight]))
            # Preparando entradas dos neuronios da camada de saida
            inputs_out.append(neurons[i].output)

        # Neuronios da camada de saida
        neurons_out = []
        for i in range(self.qntd_out):
            # Criando neuronios da camada de saida
            neurons_out.append(Neuron(inputs_out, [self.weight, self.weight, self.weight, self.weight]))
            print("neurons_out.output[",i,"] = ", neurons_out[i].output)
