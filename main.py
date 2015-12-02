#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from file_manager import *
import copy
from mlp import *
from matrix import *

def main():
    weight = 0.5    # Pesos sinaptico inicial
    mi = 0.2        # Potencial de ativacao
    n = 0.5         # Taxa de aprendizagem

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
    print("Matriz de entrada: ", end='')
    Matrix.print_matrix(examples)

    mlp = MLP()
    for line in examples:
        mlp.initialize(line, [weight, weight, weight, weight], mi)

if __name__ == '__main__':
    main()
