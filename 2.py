import numpy as np
import pandas as pd
from numpy._typing import NDArray
from fractions import Fraction
from tabulate import tabulate


def to_fraction(x):
    return str(Fraction(x).limit_denominator())


def print_simplex_tableau(tableau, var_names, index_names):
    df = pd.DataFrame(tableau, columns=var_names, index=index_names)
    df_fraction = df.map(to_fraction)
    print(tabulate(df_fraction, headers='keys', tablefmt='grid'))


def simplex_method(c: NDArray, A: NDArray, b: NDArray):
    # Вспомогательные переменные
    num_vars = len(c)
    num_constraints = len(b)

    # Построение симплекс-таблицы
    c = np.hstack([c, np.zeros(num_constraints + 1)])
    tableau = np.hstack([A, np.eye(num_constraints), b.reshape(-1, 1)])
    tableau = np.vstack([tableau, c])

    var_names = (
        ["x" + str(i)
         for i in range(1, num_vars + num_constraints + 1)] + ["b"])

    index_names = [
        "x" + str(i)
        for i in range(num_constraints + 1, num_constraints + tableau.shape[0])
    ] + ["Z"]

    while True:
        # Вывод симплекс-таблицы в начале итерации
        print_simplex_tableau(tableau, var_names, index_names)

        # Проверка на оптимальность
        if np.all(tableau[-1, :-1] >= 0):
            break

        # Определение разрешающей колонки
        pivot_col = np.argmin(tableau[-1, :-1])

        # Вычисление отношений для определения разрешающей строки
        ratios = np.array([
            tableau[i, -1] /
            tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf
            for i in range(tableau.shape[0] - 1)
        ])

        # Проверка на наличие допустимых отношений (исключая np.inf)
        valid_ratios = ratios[ratios != np.inf]

        # Если нет допустимых отношений, это может указывать на неограниченность задачи
        if valid_ratios.size == 0:
            print("Задача может быть неограниченной.")
            break

        # Определение разрешающей строки
        pivot_row = np.argmin(ratios)

        # Производим замену в базисе
        index_names[pivot_row] = "x" + str(pivot_col + 1)

        # Выбор разрешающего коэффициента
        pivot_element = tableau[pivot_row, pivot_col]

        # Пересчитываем элементы разрешающей строки
        tableau[pivot_row, :] /= pivot_element

        # Пересчитываем элементы остальных строк
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Определение решения
    x = np.zeros(num_vars + num_constraints)
    for i in range(num_vars):
        variable_index = int(index_names[i][1:]) - 1
        variable_value = tableau[i, -1]
        x[variable_index] = variable_value

    return x


# Тестовый пример
c = np.array([1, -1, -3])
A = np.array([[2, -1, 1], [-4, 2, -1], [3, 0, 1]])
b = np.array([1, 2, 5])

solution = simplex_method(c, A, b)
print("Решение: ", solution)
