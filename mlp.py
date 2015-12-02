#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import *

class MLP(object):
    qntd_out = 3        # Quantidade de saidas
    neurons_in = []        # Neuronios da camada escondida
    neurons_out = []    # Neuronios da camada de saida

    def initialize(self, inputs, weights, mi):
        inputs_out = []
        for i in range(len(inputs)):
            # Criando neuronios da camada escondida
            self.neurons_in.append(Neuron(inputs, weights, mi))
            # Preparando entradas dos neuronios da camada de saida
            inputs_out.append(self.neurons_in[i].output)

        for i in range(self.qntd_out):
            # Criando neuronios da camada de saida
            self.neurons_out.append(Neuron(inputs_out, weights, mi))

    def calculate_output(self, outputs, n):
        out_error = []
        in_error = []

        for i in range(len(self.neurons_out)):
            out_error.append(outputs[i] - self.neurons_out[i].output)

        for i in range(len(self.neurons_in)):
            error = 0
            for j in range(len(self.neurons_out)):
                error += out_error[j] * self.neurons_out[j].weight[i]

            in_error.append(error)
