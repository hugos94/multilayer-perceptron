#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from file_manager import *
import copy
from neuron import *
from matrix import *

def main():
    weight = 0.5

    fm = FileManager()

	# Le os dados de entrada a partir de um arquivo csv
    file_content = fm.read_csv("Treinamento.csv")

    # Lista de atributos do arquivo
    attributes = Matrix.extract_attributes(file_content)

    # Seleciona quantidade de linhas a serem utilizadas
    examples = Matrix.get_rows_matrix(file_content, 0, 2)

    # Remove colunas com as saidas esperadas
    examples = Matrix.remove_columns_2(examples, [4,5,6])

    # Converte elementos da matriz em float
    examples = Matrix.to_float(examples)

    # Imprime matriz a ser utilizada
    Matrix.print_matrix(examples)

    for line in examples:
        neuron = Neuron(line, [weight, weight, weight, weight])
        print("sum_inputs = ", neuron.sum_inputs())
        print("neuron.output = ", neuron.output)

if __name__ == '__main__':
    main()
