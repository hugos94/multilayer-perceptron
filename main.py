#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from file_manager import *
import copy
from mlp import *
from matrix import *

def main():
    n = 0.5         # Taxa de aprendizagem
    epoch = 5     # Quantidade de epocas

    fm = FileManager()

	# Le os dados de entrada a partir de um arquivo csv
    file_content = fm.read_csv("Treinamento.csv")

    # Remove a lista de atributos do arquivo
    attributes = Matrix.extract_attributes(file_content)

    # Seleciona quantidade de linhas a serem utilizadas
    file_content = Matrix.get_rows_matrix(file_content, 0, 5)

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

    e = epoch

    while e > 0:   #""" (DO WHILE) OU ESTA NA FAIXA DE ERRO ACEITO """
        e -= 1

        for i,line in enumerate(inputs):
            if e == (epoch - 1) and i == 0:
                # Utiliza primeira linha das entradas como inicializacao do MLP
                mlp = MLP(inputs[i])
            else:
                # Executa a rede MLP
                mlp.execute(line)

            #""" TESTAR SE SAIDA EH IGUAL A ESPERADA """
            # Calcula o erro
            if not mlp.error():
                mlp.update_weights(outputs[i], n)

            if e == 0:
                print("---: NEURONIO", i, ":---")
                for neu in mlp.neurons_out:
                    print(neu.output)
                print()

                # for i,inp in enumerate(mlp.neurons_in):
                #     print("Neuronio da escondida (", i, ")\n", inp)
                # for o,out in enumerate(mlp.neurons_out):
                #     print("Neuronios de saida (", o, ")\n", out)


if __name__ == '__main__':
    main()
