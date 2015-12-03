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

    def recalculate_output(self):
        """ Recalcula o valor de saida do neuronio """
        self.output = self.calculate_sigmoid()


    def __str__(self):
        """Retorna uma representacao em string do neuronio"""
        neuron_str = "--------------------------------------------------\n"
        neuron_str += "Potencial = "
        neuron_str += str(self.mi)
        neuron_str += "\nEntradas = | "

        for inp in self.input:
            neuron_str += str(inp)
            neuron_str += " | "
        neuron_str += "\nPesos = | "

        for wei in self.weight:
            neuron_str += str(wei)
            neuron_str += " | "
        neuron_str += "\n"

        neuron_str += "Saida = "
        neuron_str += str(self.output)
        neuron_str += "\n--------------------------------------------------\n"

        return neuron_str
