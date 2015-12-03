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
                weights_in.append(random.random())

            mi_in = random.random()
            self.neurons_in.append(Neuron(inputs, weights_in, mi_in))
            inputs_out.append(self.neurons_in[i].output)

        # Criando neuronios da camada de saida
        for i in range(architecture[2]):
            weights_out = []
            for j in range(architecture[1]):
                weights_out.append(random.random())

            mi_out = random.random()
            self.neurons_out.append(Neuron(inputs_out, weights_out, mi_out))


    def execute(self, inputs):
        for inp in self.neurons_in:
            inp.input = copy.deepcopy(inputs)
            inp.recalculate_output()


    def update_weights(self, outputs, n):
        out_error = []
        in_error = []

        for i in range(len(self.neurons_out)):
            out_error.append(outputs[i] - self.neurons_out[i].output)

        for i in range(len(self.neurons_in)):
            error = 0
            for j in range(len(self.neurons_out)):
                error += out_error[j] * self.neurons_out[j].weight[i]

            in_error.append(error)

        for i,inp in enumerate(self.neurons_in):
            for j,wei in enumerate(inp.weight):
                self.neurons_in[i].weight[j] = wei + (n * in_error[i] * inp.output)

        for i,out in enumerate(self.neurons_out):
            for j,wei in enumerate(out.weight):
                self.neurons_out[i].weight[j] = wei + (n * out_error[i] * out.output)
