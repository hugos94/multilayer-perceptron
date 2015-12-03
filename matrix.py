#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy

class Matrix:
    """Classe com operacoes de manipulacao de matrizes"""

    @staticmethod
    def remove_columns(matrix, columns):
        """
        Remove as colunas indicadas na lista columns.
        A matriz original e alterada
        """
        columns.reverse()
        for line in matrix:
            for c in columns:
                line.pop(c)


    @staticmethod
    def remove_columns_2(matrix, columns):
        """
        Remove as colunas indicadas na lista columns.
        A matriz original nao e alterada
        """
        new_matrix = []
        for line in	matrix:
            new_line = []
            for c, column in enumerate(line):
                if not c in columns:
                    new_line.append(column)
            if len(new_line) != 0:
                new_matrix.append(new_line)
        return new_matrix


    @staticmethod
    def extract_attributes(matrix):
        """
        Remove a primeira linha, que contem os nomes dos atributos
        retornando-os na forma de uma lista. A matriz original e alterada
        """
        attributes = matrix[0]
        matrix.pop(0)
        return attributes


    @staticmethod
    def extract_attributes_2(matrix):
        """
        Remove a primeira linha, que contem os nomes dos atributos
        retornando-os na forma de uma lista. A matriz original nao e alterada
        """
        attributes = matrix[0]
        return attributes


    @staticmethod
    def get_attributes(matrix):
        """
        Retorna a primeira linha, que contem os nomes dos atributos
        retornando-os na forma de uma lista. A matriz original nao e alterada
        """
        attributes = matrix[0]
        return attributes


    @staticmethod
    def print_matrix(matrix):
        """Imprime uma matriz bidimensional"""
        print("")
        for line in matrix:
            print(line)


    @staticmethod
    def get_rows_matrix(matrix, ind_in, ind_out=-1):
        """
        Retorna uma matriz comecando do ind_in e
        terminando no ind_out. A matriz original nao e alterada
        """
        if ind_out < 0:
            # Retorna a matriz inteira
            ind_out = len(matrix)
        else:
            # Retorna a matriz ate onde foi especificado
            ind_out += 1

        matrix_out = []

        for i in range(ind_in, ind_out):
            matrix_out.append(matrix[i])

        return matrix_out

    @staticmethod
    def get_rows_matrix_2(matrix, rows):
        """
        Retorna uma matriz comecando da linha ind_in e
        terminando na linha ind_out. A matriz original e alterada
        """
        rows.reverse()
        for r in rows:
            for i in range(len(matrix)):
                if r == i:
                    matrix.pop(r)

        return matrix


    @staticmethod
    def to_float(matrix):
        """
        Converte os elementos da matriz em float.
        A matriz original e alterada
        """
        for i,line in enumerate(matrix):
            for j,value in enumerate(line):
                line[j] = float(value)
            matrix[i] = line

        return matrix
