import numpy as np
from numpy._typing import NDArray

from simplex_method import *


def simplex_method_(tableau: NDArray, var_names, basis, iterations):
    while True:
        # Вывод симплекс-таблицы в начале итерации
        print_tableau(tableau, var_names, basis[:-1], iterations)

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
    X = np.zeros(len(var_names))
    for i in range(len(basis) - 2):
        variable_index = int(basis[i][1:]) - 1
        variable_value = tableau[i][-1]
        X[variable_index] = variable_value

    X_ = np.array(tableau[-1])

    Z = tableau[-1][-1]

    return X, X_, Z, iterations


def m_method(c: NDArray, A: NDArray, b: NDArray):
    # Вспомогательные переменные
    num_vars = len(c)
    num_constraints = len(b)

    # Построение симплекс-таблицы
    c = np.hstack([c, np.zeros(num_constraints + 1)])
    M = np.hstack(
        [A.sum(axis=0) * -1,
         np.zeros(num_constraints),
         b.sum(axis=0) * -1])
    tableau = np.hstack([A, np.eye(num_constraints), b.reshape(-1, 1)])
    tableau = np.vstack([tableau, c, M])

    var_names = ([f"x{i + 1}" for i in range(num_vars)] +
                 [f"y{i + 1}" for i in range(num_constraints)] + ["b"])

    basis = [f"y{i + 1}" for i in range(num_constraints)] + ["Z", ""]

    iterations = []
    while True:
        # Вывод таблицы в начале итерации
        print_tableau(tableau, var_names, basis, iterations)

        # Проверка на оптимальность
        if np.all(tableau[-1, :-1] >= 0):
            if np.all(tableau[-1, :-1] < 1e-9):
                tableau = np.delete(tableau, -1, axis=0)
                return simplex_method_(tableau, var_names, basis, iterations)
            elif np.any(tableau[-1] > 0):
                print("Нет допустимого решения, так как искусственные переменные остались в базисе.")
                return [], [], None, []

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

        # Удаляем колонку, отвечающую за искусственную переменную
        if basis[pivot_row].startswith('y'):
            tableau = np.delete(tableau,
                                var_names.index(basis[pivot_row]),
                                axis=1)
            var_names.remove(basis[pivot_row])

        # Производим замену в базисе
        basis[pivot_row] = f"x{pivot_col + 1}"

        # Выбор разрешающего коэффициента
        pivot_element = tableau[pivot_row, pivot_col]

        # Пересчитываем элементы разрешающей строки
        tableau[pivot_row, :] /= pivot_element

        # Пересчитываем элементы остальных строк
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]


def run_m_method_analysis(c: NDArray, A: NDArray, b: NDArray):
    # Решение тестового примера м-методом
    X, X_, Z, iterations = m_method(c, A, b)

    # Проверка на наличие решения
    if len(X) == 0:
        print("Решение отсутствует")
        return

    # Получение строки итераций метода для HTML документа
    method_iterations = "".join(iterations)

    # Создание таблицы оптимального решения для HTML документа
    solution_table = create_solution_table(X, Z).to_html(index=False)

    # Создание таблицы статуса ресурсов для HTML документа
    resource_status_table = create_resource_status_table(
        X, b).to_html(index=False)

    # Создание и корректировка таблицы ценности ресурса для HTML документа
    resource_value_table = create_resource_value_analysis_table(
        X_, b).to_html(index=False)
    resource_value_table = adjust_resource_value_table(resource_value_table)

    # Сохранение результатов в HTML документ
    write_html_document(method_iterations, solution_table,
                        resource_status_table, resource_value_table, 'templates/m_method.html')


if __name__ == "__main__":
    # Тестовый пример
    """
        5x_1 + 3x_2 + 4x_3 - x_4 = Z

        x_1 + 3x_2 + 2x_3 + 2x_4 = 3
        2x_1 + 2x_2 + x_3 + x_4  = 3
    """
    c = np.array([-5, -3, -4, 1])
    A = np.array([[1, 3, 2, 2], [2, 2, 1, 1]])
    b = np.array([3, 3])
    """
        x_1 + x_2 = Z
        
        2x_1 + x_2 - x_3 = 4
        x_1 + 2x_2 = 6
    """
    c = np.array([-1, -1, 0])
    A = np.array([[2, 1, -1], [1, 2, 0]])
    b = np.array([4, 6])

    solution = m_method(c, A, b)
    print("Решение: ", solution)
