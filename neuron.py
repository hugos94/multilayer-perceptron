#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import copy

class Neuron(object):
    """Construtor da classe Neuron"""

    def __init__(self, weights, theta):

        # Inicializa as entradas do neuronio com uma lista vazia
        self.input = []

        # Lista de pesos aleatorios para cada entrada (Entre 0 e 1)
        self.weight = []
        self.weight = copy.deepcopy(weights)

        # Limiar de ativacao do neuronio
        self.theta = theta

        # Saida do neuronio com zero
        self.output = 0.0

    def calculate_sigmoid(self):
        """ Calcula a funcao sigmoide """

        beta = 1
        potencial = self.sum_inputs() - self.theta

        e = math.exp(-1 * beta * potencial)  # e^-1(x)
        out = 1 / (1 + e)

        return out


    def calculate_derived_sigmoid(self):
        """ Calcula a derivada da funcao sigmoide """

        beta = 1
        potencial = self.sum_inputs() - self.theta

        e_1 = math.exp((-1 * beta * potencial) - 1)  # e^-1(x)
        out = beta * potencial * e_1

        e_2 = math.exp(-1 * beta * potencial)  # e^-1(x)
        out = out / math.pow((1 + e_2), 2)

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
        neuron_str += str(self.theta)
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
