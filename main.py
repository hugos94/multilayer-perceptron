#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from file_manager import *
import copy
from neuron import *

def main():
    fm = FileManager()

	# Le os dados de entrada a partir de um arquivo csv
	file_content = fm.read_csv("Treinamento.csv")

	# Clona os dados de entrada
	examples = copy.deepcopy(file_content)

    neuron = NEURON([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5])
    print(neuron.sum_inputs())
    print(neuron.output)

if __name__ == '__main__':
    main()
