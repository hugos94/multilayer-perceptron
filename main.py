#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from file_manager import *
import copy
from mlp import *
from matrix import *

def main():
    n = 0.5         # Taxa de aprendizagem
    epoch = 0     # Quantidade de epocas

    fm = FileManager()

	# Le os dados de entrada a partir de um arquivo csv
    file_content = fm.read_csv("Treinamento.csv")

    # Lista de atributos do arquivo
    attributes = Matrix.extract_attributes(file_content)

    # Seleciona quantidade de linhas a serem utilizadas
    file_content = Matrix.get_rows_matrix(file_content, 0, 0)

    # Remove colunas com as saidas esperadas
    inputs = Matrix.remove_columns_2(file_content, [4,5,6])

    # Devolve colunas com as saidas esperadas
    outputs = Matrix.remove_columns_2(file_content, [0,1,2,3])

    # Converte elementos das matrizes em float
    inputs = Matrix.to_float(inputs)
    outputs = Matrix.to_float(outputs)

    # Imprime matriz a ser utilizada
    print("Matriz de entrada: ", end='')
    Matrix.print_matrix(inputs)

    # Utiliza primeira linha das entradas como inicializacao do MLP
    mlp = MLP(inputs[0])
    Matrix.get_rows_matrix_2(inputs, [0])

    for i,inp in enumerate(mlp.neurons_in):
        print("Neuronio da escondida (", i, ")\n", inp)

    for o,out in enumerate(mlp.neurons_out):
        print("Neuronios de saida (", o, ")\n", out)
    # while epoch >= 0:
    #     epoch -= 1
    #
    #     for i,line in enumerate(inputs):
            # Executa a rede MLP
            #mlp.execute(line)

            # Calcula o erro
            #mlp.calculate_output(outputs[i], n)

if __name__ == '__main__':
    main()
