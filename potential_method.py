from typing import Any, List, Set, Tuple, Literal

import numpy as np
from numpy.typing import NDArray

from northwest_corner_method import balance_supply_demand, northwest_corner_method


def find_potentials(cost: NDArray[Any], basis_indices: List[Tuple[int, int]]):
    m, n = cost.shape
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0

    while np.isnan(u).any() or np.isnan(v).any():
        for i, j in basis_indices:
            if np.isnan(u[i]) and not np.isnan(v[j]):
                u[i] = cost[i, j] - v[j]
            elif not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = cost[i, j] - u[i]

    return u, v


def check_optimality(cost: NDArray[Any], basis_indices: List[Tuple[int, int]], u: NDArray[Any], v: NDArray[Any]):
    m, n = cost.shape
    for i in range(m):
        for j in range(n):
            if (i, j) not in basis_indices:
                if (u[i] + v[j]) > cost[i, j]:
                    return False
    return True


def choose_start_cell(cost, basis_indices: List[Tuple[int, int]], u: NDArray[Any], v: NDArray[Any]):
    m, n = cost.shape
    deltas = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            if (i, j) not in basis_indices:
                if (u[i] + v[j]) > cost[i, j]:
                    deltas[i, j] = (u[i] + v[j]) - cost[i, j]
    
    max_delta = np.max(deltas)
    max_delta_cells = np.argwhere(deltas == max_delta)
    min_cost_cells = sorted(max_delta_cells, key=lambda pos: cost[pos[0], pos[1]])
    start_cell = tuple(min_cost_cells[0])

    return start_cell


def find_cycle(basis_indices: List[Tuple[int, int]], start_cell):
    visited: Set[Tuple[int, int]] = set()  # Для отслеживания посещенных ячеек
    path = []  # Путь цикла

    def search(curr_cell, prev_cell, type_of_movement: Literal['', 'vertical', 'horizontal']):
        if curr_cell in visited:
            if curr_cell == start_cell and len(path) > 3:  # Цикл найден, должен быть длиннее 3 шагов
                return True
            return False

        visited.add(curr_cell)
        path.append(curr_cell)

        # Проверяем соседние базисные ячейки
        for next_cell in (basis_indices + [start_cell]):
            if next_cell == prev_cell:  # Игнорируем ячейку, из которой пришли
                continue

            next_cell_row, next_cell_col = next_cell
            curr_cell_row, curr_cell_col = curr_cell

            if next_cell_row == curr_cell_row and type_of_movement != 'horizontal':
                if search(next_cell, curr_cell, 'horizontal'):
                    return True

            if next_cell_col == curr_cell_col and type_of_movement != 'vertical':
                if search(next_cell, curr_cell, 'vertical'):
                    return True

        visited.remove(curr_cell)
        path.pop()
        return False

    if search(start_cell, None, ''):
        return path
    else:
        return []


def redistribute_load(X, cost, basis_indices: List[Tuple[int, int]], cycle: List[Tuple[int, int]]):
    min_weight = float('inf')
    for i, j in cycle[1::2]:
        min_weight = min(min_weight, X[i][j])

    min_weight_cells =  []
    for i, j in cycle[1::2]:
        if X[i][j] == min_weight:
            min_weight_cells.append((i, j))

    max_cost_cell = tuple(sorted(min_weight_cells, key=lambda pos: cost[pos[0]][pos[1]], reverse=True)[0])

    basis_indices.remove(max_cost_cell)
    basis_indices.append(cycle[0])

    for i, j in cycle[::2]:
        X[i][j] += min_weight

    for i, j in cycle[1::2]:
        X[i][j] -= min_weight

    return X


def potential_method(supply: np.ndarray, demand: np.ndarray, cost: np.ndarray):
    # Определяем допустимое базисное решение
    X, basis_indices = northwest_corner_method(supply, demand)

    while True:
        # Шаг 1: построение системы потенциалов
        u, v = find_potentials(cost, basis_indices)

        # Шаг 2: проверка оптимальности найденного решения
        is_optimal = check_optimality(cost, basis_indices, u, v)
        if is_optimal:
            break

        # Шаг 3: выбор клетки, в которую необходимо послать перевозку
        start_cell = choose_start_cell(cost, basis_indices, u, v)

        # Шаг 4: построение нового допустимого базисного решения
        cycle = find_cycle(basis_indices, start_cell)
        X = redistribute_load(X, cost, basis_indices, cycle)

    return X


if __name__ == "__main__":
    suppliers = [
        np.array([100, 250, 200, 300]),
        np.array([200, 175, 225]),
        np.array([200, 450, 250]),
        np.array([250, 200, 200]),
        np.array([350, 330, 270]),
        np.array([300, 250, 200]),
        np.array([350, 200, 300]),
        np.array([200, 250, 200]),
        np.array([230, 150, 170]),
        np.array([200, 300, 250]),
        np.array([200, 350, 300])
    ]
    demanders = [
        np.array([200, 200, 100, 100, 250]),
        np.array([100, 125, 325, 250, 100]),
        np.array([100, 125, 325, 250, 100]),
        np.array([120, 130, 100, 160, 110]),
        np.array([210, 170, 220, 150, 200]),
        np.array([210, 170, 220, 150, 200]),
        np.array([170, 140, 200, 195, 145]),
        np.array([190, 100, 120, 110, 130]),
        np.array([140, 90, 160, 110, 150]),
        np.array([210, 150, 120, 135, 135]),
        np.array([270, 130, 190, 150, 110])
    ]
    costs = [
        np.array([[10, 7, 4, 1, 4], [2, 7, 10, 6, 11], [8, 5, 3, 2, 2], [11, 8, 12, 16, 13]]),
        np.array([[5, 7, 4, 2, 5], [7, 1, 3, 1, 10], [2, 3, 6, 8, 7]]),
        np.array([[5, 8, 7, 10, 3], [4, 2, 2, 5, 6], [7, 3, 5, 9, 2]]),
        np.array([[27, 36, 35, 31, 29], [22, 23, 26, 32, 35], [35, 42, 38, 32, 39]]),
        np.array([[3, 12, 9, 1, 7], [2, 4, 11, 2, 10], [7, 14, 12, 5, 8]]),
        np.array([[4, 8, 13, 2, 7], [9, 4, 11, 9, 17], [3, 16, 10, 1, 4]]),
        np.array([[22, 14, 16, 28, 30], [19, 17, 26, 36, 36], [37, 30, 31, 39, 41]]),
        np.array([[28, 27, 18, 27, 24], [18, 26, 27, 32, 21], [27, 33, 23, 31, 34]]),
        np.array([[40, 19, 25, 26, 35], [42, 25, 27, 15, 38], [46, 27, 36, 40, 45]]),
        np.array([[20, 10, 12, 13, 16], [25, 19, 20, 14, 10], [17, 18, 15, 10, 17]]),
        np.array([[24, 50, 45, 27, 15], [20, 32, 40, 35, 30], [22, 16, 18, 28, 20]])
    ]

    for i, (supply, demand, cost) in enumerate(zip(suppliers, demanders, costs)):
        print(f"#{i}")

        supply, demand, cost = balance_supply_demand(supply, demand, cost)
        optimal_plan = potential_method(supply, demand, cost)

        components_count = np.count_nonzero(optimal_plan)
        if components_count == (supply.shape[0] + demand.shape[0] - 1):
            print("Найденное решение является базисным")
            print("Найденное решение является невырожденным")
        else:
            print("Найденное решение не является базисным")
            print("Найденное решение является вырожденным")

        print(f"Допустимое решение: \n{optimal_plan}")

        supplies = np.argwhere(optimal_plan > 0)
        for i, j in supplies:
            print(f"Поставщик {i + 1} поставил {optimal_plan[i, j]} груза {j + 1} потребителю")

        Z = np.sum(optimal_plan * cost)
        print(f"Начальное значение функции: {Z}\n")
