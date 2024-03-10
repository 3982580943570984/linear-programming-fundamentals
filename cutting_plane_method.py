from typing import List, Tuple
from fractions import Fraction
from math import modf

import numpy as np
import pandas as pd

from sympy import symbols, Eq, latex

import globals as g
from dual_simplex_method import dual_simplex_method
from utilities import selective_rounding, to_canonical


def m_method(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = np.hstack([-A.sum(axis=0), np.zeros(len(b))])
    A = np.hstack([A, np.eye(len(b))])
    c = np.hstack([c, np.zeros(len(b))])

    while True:
        # Запоминаем состояние текущей итерации
        g.m_method_iterations.append((A.copy(), b.copy(), c.copy(), M.copy()))

        # Шаг 1: проверка базисного решения на оптимальность
        M = selective_rounding(M)
        if np.all(M[:-A.shape[0]] == 0) and np.all(M[-A.shape[0]:] == 1):
            return A[:, :-A.shape[0]], b, c[:-A.shape[0]]
        elif np.all(M > 0):
            raise ValueError("Нет допустимого решения, так как искусственные переменные остались в базисе.")

        # Шаг 2: определение разрешающего столбца
        p = np.argmin(M)

        # Шаг 3: проверка признака неограниченности целевой функции
        if np.all(A[:, p] <= 0):
            raise ValueError("Целевая функция неограниченна.")

        # Шаг 4: вычисление симплексных отношений
        s = b / A[:, p]
        s[A[:, p] <= 0] = np.inf

        # Шаг 5: определение разрешающей строки
        q = np.argmin(s)

        # Шаг 6: пересчет таблицы
        b[q] /= A[q, p]
        A[q] /= A[q, p]
        for i in range(A.shape[0]):
            if i != q:
                b[i] -= A[i, p] * b[q]
                A[i] -= A[i, p] * A[q]
        c -= c[p] * A[q]
        M -= M[p] * A[q]


def create_m_method_interpretations():
    interpretations = []
    for iteration in g.m_method_iterations:
        A, b, c, M = iteration
        
        # Строим таблицу
        tableau = np.vstack([
            np.hstack([A, b.reshape(-1, 1)]),
            np.hstack([c, [0]]),
            np.hstack([M, [0]])
        ])
        
        # Определяем названия столбцов
        columns: List[str] = ([f"x{i}" for i in range(A.shape[1])])
        columns.append('b')

        # Определяем названия строк
        indices: List[str] = []
        for i in range(A.shape[1]):
            column = A[:, i]
            if np.count_nonzero(column) == 1 and column.sum() == 1 and c[i] == 0:
                indices.append(f'x{i}')
        indices.append('Z')
        indices.append('M')

        # Переводим таблицу в HTML представление
        df = pd.DataFrame(tableau, columns=columns, index=indices).map(lambda x: str(Fraction(x).limit_denominator())).to_html(index=True) #type: ignore

        interpretations.append(f"<h2>Итерация М-метода {len(interpretations)}</h2>" + df)
    return interpretations


def cutting_plane_method(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # Определяем наличие начального базиса
    count = np.sum((np.vstack([A, c]).sum(axis=0) == 1) & (np.count_nonzero(np.vstack([A, c]), axis=0) == 1))
    if count != A.shape[0]:
        A, b, c = m_method(A, b, c)

    while True:
        # Запоминаем состояние текущей итерации
        g.cutting_plane_method_iterations.append((A.copy(), b.copy(), c.copy()))
        g.cutting_plane_method_equations.append('')

        # Шаг 1: нахождение оптимального решения задачи без учета условия целочисленности
        X = dual_simplex_method(A, b, c)
        X = selective_rounding(X)

        # Шаг 2: проверка целочисленности оптимального решения
        fractional_parts, _ = np.modf(X)
        contains_fractional_part = np.any(fractional_parts != 0)
        if not contains_fractional_part:
            return X

        # Шаг 3: составление дополнительного ограничения, которое отсекает нецелочисленные решения
        equation_index = np.argmax(fractional_parts)
        equation_fractional_parts, _ = np.modf(A[equation_index])
        contains_fractional_part = np.any(equation_fractional_parts != 0)
        if not contains_fractional_part:
            raise ValueError("Целочисленное решение отсутствует.")

        g.cutting_plane_method_equations[-1] = generate_equation(equation_fractional_parts, -modf(b[equation_index])[0])

        column = np.zeros(A.shape[0])
        column = np.hstack([column, [1]])
        column = column.reshape(-1, 1)

        A = np.vstack([A, [-equation_fractional_parts]])
        A = np.hstack([A, column])
        b = np.hstack([b, [-modf(b[equation_index])[0]]])
        c = np.hstack([c, [0]])


def create_cutting_plane_interpretations():
    interpretations = []
    for iteration in g.cutting_plane_method_iterations:
        A, b, c = iteration
        
        # Строим таблицу
        tableau = np.vstack([
            np.hstack([A, b.reshape(-1, 1)]),
            np.hstack([c, [0]])
        ])
        
        # Определяем названия столбцов
        columns: List[str] = ([f"x{i}" for i in range(A.shape[1])])
        columns.append('b')

        # Определяем названия строк
        indices: List[str] = []
        for i in range(A.shape[1]):
            column = A[:, i]
            if np.count_nonzero(column) == 1 and column.sum() == 1 and c[i] == 0:
                indices.append(f'x{i}')
        indices.append('Z')

        # Переводим таблицу в HTML представление
        df = pd.DataFrame(tableau, columns=columns, index=indices).map(lambda x: str(Fraction(x).limit_denominator())).to_html(index=True) #type: ignore

        interpretations.append(f"<h2>Итерация метода Гомори {len(interpretations)}</h2>" + df)
    return interpretations


def generate_equation(equation_fractional_parts, right_side) -> str:
    equation_fractional_parts = selective_rounding(equation_fractional_parts)
    
    x = symbols(f'x1:{len(equation_fractional_parts)+1}')
    left_side = sum([-coeff * var for coeff, var in zip(equation_fractional_parts, x)])
    
    equation = latex(Eq(left_side, np.around(right_side, decimals=1)))
    
    return f'$${equation}$$'


if __name__ == "__main__":
    A = np.array([[5, 2], [8, 4]]).astype(float)
    b = np.array([20, 38]).astype(float)
    c = np.array([7, 3]).astype(float)
    signs = ['<=', '<=']
    A, b, c = to_canonical(A, b, c, signs)
    print(cutting_plane_method(A, b, c))

    A = np.array([[1, 3, 2, 2], [2, 2, 1, 1]]).astype(float)
    b = np.array([3, 3]).astype(float)
    c = np.array([5, 3, 4, -1]).astype(float)
    signs = ['=', '=']
    A, b, c = to_canonical(A, b, c, signs)
    print(cutting_plane_method(A, b, c))

    A = np.array([[2, 1], [1, 2]]).astype(float)
    b = np.array([4, 6]).astype(float)
    c = np.array([1, 1]).astype(float)
    signs = ['>=', '=']
    A, b, c = to_canonical(A, b, c, signs)
    print(cutting_plane_method(A, b, c))
