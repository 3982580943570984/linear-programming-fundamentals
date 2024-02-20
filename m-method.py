import numpy as np
import pandas as pd
from numpy._typing import NDArray
from fractions import Fraction
from tabulate import tabulate


def to_fraction(x):
    return str(Fraction(x).limit_denominator())


def print_tableau(tableau, var_names, index_names):
    df = pd.DataFrame(tableau, columns=var_names, index=index_names)
    df_fraction = df.map(to_fraction)
    print(tabulate(df_fraction, headers='keys', tablefmt='grid'))


def simplex_method(tableau: NDArray, var_names: list[str], index_names: list[str]):
    while True:
        # Вывод симплекс-таблицы в начале итерации
        print_tableau(tableau, var_names, index_names)

        # Проверка на оптимальность
        if np.all(tableau[-1, :-1] >= 0):
            print("Optimal")
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
    x = np.zeros(len(var_names))
    for i in range(len(var_names)):
        variable_index = int(index_names[i][1:]) - 1
        variable_value = tableau[i, -1]
        x[variable_index] = variable_value

    return x


def m_method(c: NDArray, A: NDArray, b: NDArray):
    # Вспомогательные переменные
    num_vars = len(c)
    num_constraints = len(b)
    num_artifical = len(b)

    # Построение симплекс-таблицы
    c = np.hstack([c * -1, np.zeros(num_constraints + 1)])
    M = np.hstack(
        [A.sum(axis=0) * -1,
         np.zeros(num_artifical),
         b.sum(axis=0) * -1])
    tableau = np.hstack([A, np.eye(num_constraints), b.reshape(-1, 1)])
    tableau = np.vstack([tableau, c, M])

    var_names = (["x" + str(i) for i in range(1, num_vars + 1)] +
                 ["y" + str(i) for i in range(1, num_constraints + 1)] + ["b"])

    basis = ["y" + str(i) for i in range(1, num_constraints + 1)] + ["Z", ""]

    while True:
        # Вывод таблицы в начале итерации
        print_tableau(tableau, var_names, basis)

        # Проверка на оптимальность
        if np.all(tableau[-1, :-1] >= 0):
            if np.all(tableau[-1, :-1] < 1e-9):
                tableau = np.delete(tableau, -1, axis=0)
                return simplex_method(tableau, var_names, basis)
            elif np.any(tableau[-1] > 0):
                print("Нет допустимого решения, так как искусственные переменные в базисе.")
                return None

        # Определение разрешающей колонки
        pivot_col = np.argmin(tableau[-1, :-1])

        # Вычисление отношений для определения разрешающей строки
        ratios = np.array([
            tableau[i, -1] /
            tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf
            for i in range(tableau.shape[0] - 1)
        ])

        print("ratios: ", ratios)

        # Проверка на наличие допустимых отношений (исключая np.inf)
        valid_ratios = ratios[ratios != np.inf]

        # Если нет допустимых отношений, это может указывать на неограниченность задачи
        if valid_ratios.size == 0:
            print("Задача может быть неограниченной.")
            break

        # Определение разрешающей строки
        pivot_row = np.argmin(ratios)

        if basis[pivot_row][0] == 'y':
            tableau = np.delete(tableau,
                                var_names.index(basis[pivot_row]),
                                axis=1)
            var_names.remove(basis[pivot_row])

        # Производим замену в базисе
        basis[pivot_row] = "x" + str(pivot_col + 1)

        # Выбор разрешающего коэффициента
        pivot_element = tableau[pivot_row, pivot_col]

        # Пересчитываем элементы разрешающей строки
        tableau[pivot_row, :] /= pivot_element

        # Пересчитываем элементы остальных строк
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Определение решения
    x = np.zeros(num_vars)
    for i in range(len(basis) - 2):
        variable_index = int(basis[i][1:]) - 1
        variable_value = tableau[i, -1]
        x[variable_index] = variable_value

    return x


# Тестовый пример
c = np.array([5, 3, 4, -1])
A = np.array([[1, 3, 2, 2], [2, 2, 1, 1]])
b = np.array([3, 3])

solution = m_method(c, A, b)
print("Решение: ", solution)
