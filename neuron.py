#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import copy

class Neuron(object):
    """Construtor da classe Neuron"""
    def __init__(self, inputs, weights, mi):
        # Lista de entradas do neurônio
        self.input = []
        self.input = copy.deepcopy(inputs)

        # Lista de pesos para cada entrada
        self.weight = []
        self.weight = copy.deepcopy(weights)

        # Potencial de ativacao do neuronio
        self.mi = mi

        # Saida do neuronio
        self.output = self.calculate_sigmoid()

    def calculate_sigmoid(self):
        """ Calcula a funcao sigmoide """
        sum = self.sum_inputs()

        e = math.exp(-1 * (self.mi - sum))  # e^-1(x)
        out = 1 / (1 + e)

        return out

    def sum_inputs(self):
        """ Calcula o somatorio das entradas * pesos """
        sum = 0

        for i in range(len(self.input)):
            sum += self.input[i] * self.weight[i]

        return sum
