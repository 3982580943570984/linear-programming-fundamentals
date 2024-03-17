from typing import List
from fractions import Fraction

import numpy as np
import pandas as pd

import globals as g
from utilities import to_canonical


def simplex_method(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    while True:
        # Шаг 1: проверка текущего допустимого базисного решения
        if np.all(c >= 0):
            break

        # Шаг 2: определение разрешающего столбца
        p = np.argmin(c)

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
        for i in range(len(A)):
            if i != q:
                b[i] -= A[i, p] * b[q]
                A[i] -= A[i, p] * A[q]
        c -= c[p] * A[q]

        # Запоминаем состояние текущей итерации
        iteration_count = len(g.cutting_plane_method_iterations)
        g.simplex_method_iterations[iteration_count].append(((A.copy(), b.copy(), c.copy())))

    # Определяем решение
    X = np.zeros(A.shape[1])
    for i in range(len(X)):
        column = A[:, i]
        if np.count_nonzero(column) == 1 and column.sum() == 1 and c[i] == 0:
            row_index = np.where(column == 1)[0][0]
            X[i] = b[row_index]

    return X


def create_simplex_interpretations():
    for iteration, state in g.simplex_method_iterations.items():
        for A, b, c in state:
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

            g.interpretations[iteration].append(f"<h2>Итерация симплекс-метода</h2>" + df)


def dual_simplex_method(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    while True:
        # Шаг 1: выбор способа решения задачи
        if np.all(b >= 0):
            return simplex_method(A, b, c)

        # Шаг 2: проверка разрешимости задачи
        for i in range(len(A)):
            if b[i] < 0 and np.all(A[i] >= 0):
                raise ValueError("Исходная задача неразрешима.")

        # Шаг 3: выбор разрешающей строки
        q = np.argmin(b)

        # Шаг 4: вычисление двойственных отношений
        d = np.abs(c / A[q])
        d[A[q] >= 0] = np.inf

        # Шаг 5: выбор разрешающего столбца
        p = np.argmin(d)

        # Шаг 6: пересчет таблицы
        b[q] /= A[q, p]
        A[q] /= A[q, p]
        for i in range(A.shape[0]):
            if i != q:
                b[i] -= A[i, p] * b[q]
                A[i] -= A[i, p] * A[q]
        c -= c[p] * A[q]

        # Запоминаем состояние текущей итерации
        iteration_count = len(g.cutting_plane_method_iterations)
        g.dual_simplex_method_iterations[iteration_count].append(((A.copy(), b.copy(), c.copy())))


def create_dual_simplex_interpretations():
    for iteration, state in g.dual_simplex_method_iterations.items():
        for A, b, c in state:
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

            g.interpretations[iteration].append(f"<h2>Итерация двойственного симплекс-метода</h2>" + df)


if __name__ == "__main__":
    A = np.array([[2, -1, 1], [-4, 2, -1], [3, 0, 1]]).astype(float)
    b = np.array([1, 2, 5]).astype(float)
    c = np.array([-1, 1, 3]).astype(float)
    signs = ['<=', '<=', '<=']
    A, b, c = to_canonical(A, b, c, signs)
    print(simplex_method(A, b, c))

    A = np.array([[1, 1, -1], [1, -5, 1]]).astype(float)
    b = np.array([4, 5]).astype(float)
    c = np.array([2, -1, -5]).astype(float)
    signs = ['<=', '>=']
    A, b, c = to_canonical(A, b, c, signs)
    print(dual_simplex_method(A, b, c))
