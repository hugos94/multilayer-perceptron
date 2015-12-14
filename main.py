#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from file_manager import *
import copy
from mlp import *
from matrix import *

def main():

    fm = FileManager()

	# Le os dados de entrada a partir de um arquivo csv
    file_content = fm.read_csv("Treinamento.csv")

    # Remove a lista de atributos do arquivo
    attributes = Matrix.extract_attributes(file_content)

    # Seleciona quantidade de linhas a serem utilizadas
    file_content = Matrix.get_rows_matrix(file_content, 0, 10)

    # Devolve colunas com as entradas
    inputs = Matrix.remove_columns_2(file_content, [4,5,6])

    # Devolve colunas com as saidas esperadas
    outputs = Matrix.remove_columns_2(file_content, [0,1,2,3])

    # Converte elementos das matrizes em float
    inputs = Matrix.to_float(inputs)
    outputs = Matrix.to_float(outputs)

    # Imprime matriz a ser utilizada
    print("Matriz de entrada: ", end='')
    Matrix.print_matrix(inputs)

    learning_tax = 0.5  # Taxa de aprendizagem
    epoch = 10  # Quantidade de epocas

    mlp = MLP()

    mlp.trainning(epoch, learning_tax, inputs, outputs)

    mlp.test(inputs, outputs)

if __name__ == '__main__':
    main()
