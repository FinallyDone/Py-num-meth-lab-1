##################################################################################
#   48 Группа, Абдюшев Тимур, Лабораторная работа 1

#   Численные методы решения систем линейных алгебраических уравнений.
#   Метод Гаусса с выбором главного элемента по столбцам.
#
#   1) Решить систему линейных уравнений Ax = f с выбором главного элемента
#      по столбцам методом Гаусса.
#
#   2) Вычислить вектор невязки r = Ax - f, где x – полученное решение.
#
#   3) Вычислить определитель матрицы A используя метод Гаусса.
#
#   4) Найти обратную матрицу А^(-1) используя метод Гаусса.
#
#   5) Сделать проверку, умножить матрицу А на полученную матрицу А^(-1)
#
##################################################################################


import numpy as np
import math


# Метод Гауса с выбором главного элемента по столбцу
def method_gaus_col(matrix_A, matrix_F):
    # Определитель матрицы А
    det = 1
    # Длина столбца и строки
    len_col = len(matrix_A)
    len_row = len(matrix_A[0])
    # Берем длину строки
    for i in range(len_row):
        # Поиск максимального элемента в столбце
        max_elem = 0.0
        max_elem_index = 0
        # Берем длину столбца
        for j in range(len_col-i):
            if math.fabs(matrix_A[j+i][i]) > max_elem:
                max_elem = math.fabs(matrix_A[j+i][i])
                max_elem_index = j+i

        # Меняем 1 строку на строку с главным элементом в обоих матрицах
        if max_elem_index != 1:
            matrix_A[i], matrix_A[max_elem_index] = matrix_A[max_elem_index], matrix_A[i].copy()
            matrix_F[i], matrix_F[max_elem_index] = matrix_F[max_elem_index], matrix_F[i].copy()

        # Делим строку на главный элемент
        # Проверяем, не является ли 2 матрица вектором
        if len(matrix_F[0]) <= 1:
            matrix_F[i] /= matrix_A[i][i].copy()
        else:
            matrix_F[i:i+1, :] /= matrix_A[i][i].copy()
        matrix_A[i:i+1, :] /= matrix_A[i][i].copy()

        # Приводим к верхне-угольной матрице матрицу А
        if i != (len_col-1):
            if len(matrix_F[0]) <= 1:
                matrix_F[i+1:] -= matrix_F[i:i+1] * matrix_A[i+1:, i:i+1]
            else:
                matrix_F[i+1:, :] -= matrix_F[i:i+1, :] * matrix_A[i+1:, i:i+1]
            matrix_A[i+1:, :] -= matrix_A[i:i+1, :] * matrix_A[i+1:, i:i+1]

    # Вычитаем значения и получаем решения
    for i in range(len_col-1):
        for j in range(i+1):
            matrix_F[len_col - 2 - i] -= matrix_A[len_col-2-i][len_row-1-j] * matrix_F[len_col - 1 - j]
            matrix_A[len_col-2-i][len_row-1-j] = 0

    # Вектор решений
    return matrix_F


# Нахождение определителя методм Гаусса
def method_gaus_opred(matrix_A):
    det = 1
    # Длина столбца
    len_col = len(matrix_A)
    for i in range(len_col):
        t = i
        for j in range(i + 1, len_col):
            if abs(matrix_A[j][i]) > abs(matrix_A[t][i]):
                t = j

        matrix_A[i], matrix_A[t] = matrix_A[t], matrix_A[i]
        if i != t:
            det = -det
        det *= matrix_A[i][i]
        for j in range(i + 1, len_col):
            matrix_A[i][j] /= matrix_A[i][i]

        for j in range(len_col):
            if j != i:
                for k in range(i + 1, len_col):
                    matrix_A[j][k] -= matrix_A[i][k] * matrix_A[j][i]
    return det


# Вычисление вектора невязки
def vector_nevyazki(matrix_A, matrix_F, vector_x):
    return matrix_A.dot(vector_x) - matrix_F


# Функция печати в консоль матрицы
def print_matrix(matrix, str = '', before=8, after=4):
    # Печать числа с настройкой чисел до и после точки
    f = f'{{: {before}.{after}f}}'
    print(str)
    print('\n'.join([f''.join(f.format(el)
                    for el in row)
                    for row in matrix]) + '\n')


if __name__ == '__main__':
    matrix_A_arr = [
        [4.3, 4.2, -3.2, 9.3],
        [7.9, 5.6, 5.7, -7.2],
        [8.5, -4.8, 0.8, 3.5],
        [3.2, -1.4, 8.9, 3.3]
    ]
    matrix_A = np.array(matrix_A_arr, float)

    matrix_F_arr = [
        [8.6],
        [6.68],
        [9.95],
        [1]
    ]
    matrix_F = np.array(matrix_F_arr, float)

    # 1) Решение системы лин. ур. Ax = f с выбором
    #   глав элемента по столбцам и вывод вектора решений
    vector_x = method_gaus_col(matrix_A.copy(), matrix_F.copy())
    print_matrix(vector_x, "Вектор решения:")
    # 2) Вычисление вектора невязки
    vector_r = vector_nevyazki(matrix_A.copy(), matrix_F.copy(), vector_x.copy())
    print_matrix(vector_r, "Вектор невязки:", 8, 16)
    # 3) Вычисление определителя матрицы А методом Гаусса
    det = method_gaus_opred(matrix_A_arr)
    print("Определитель методом Гаусса:", det, "\n")
    # 4) Нахождение обратной матрицы A методом Гаусса
    matrix_At = method_gaus_col(matrix_A.copy(), np.eye(len(matrix_A)))
    print_matrix(matrix_At, "Обратная матрица:")
    # 5) Проверка, умножение матрицы А на ее обраную матрицу
    print_matrix(matrix_At.dot(matrix_A.copy()), "Умножение матрицы А на ее обратную матрицу:")
