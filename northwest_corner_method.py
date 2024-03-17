from typing import List, Tuple, Any

import numpy as np
from numpy.typing import NDArray


def balance_supply_demand(supply: NDArray[Any], demand: NDArray[Any], cost: NDArray[Any]):
    supply_total = np.sum(supply)
    demand_total = np.sum(demand)
    
    if supply_total != demand_total:
        print("Производим балансирование спроса и предложения")
        if supply_total > demand_total:
            print(f"Спрос - {supply_total} > Предложение - {demand_total}")
            demand = np.append(demand, supply_total - demand_total)
            cost = np.hstack([cost, np.zeros(supply.shape[0]).reshape(-1, 1)])
        else:
            print(f"Спрос - {supply_total} <= Предложение - {demand_total}")
            supply = np.append(supply, demand_total - supply_total)
            cost = np.vstack([cost, np.zeros(demand.shape[0])])

    return supply, demand, cost


def northwest_corner_method(supply: NDArray[Any], demand: NDArray[Any]):
    i, j = 0, 0
    m, n = len(supply), len(demand)
    X = np.zeros((m, n))
    basis_indices: List[Tuple[int, int]] = []
    while i != m and j != n:
        basis_indices.append((i, j))
        if supply[i] <= demand[j]:
            demand[j] -= supply[i]
            X[i][j] = supply[i]
            i += 1
        else:
            supply[i] -= demand[j]
            X[i][j] = demand[j]
            j += 1
    return X, basis_indices


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
        X, basis_indices = northwest_corner_method(supply.copy(), demand.copy())
        components_count = np.count_nonzero(X)
        if components_count == (supply.shape[0] + demand.shape[0] - 1):
            print("Найденное решение является базисным")
            print("Найденное решение является невырожденным")
        else:
            print("Найденное решение не является базисным")
            print("Найденное решение является вырожденным")
        print(f"Допустимое решение: \n{X}")
        supplies = np.argwhere(X > 0)
        for i, j in supplies:
            print(f"Поставщик {i + 1} поставил {X[i, j]} груза {j + 1} потребителю")
        Z = np.sum(X * cost)
        print(f"Начальное значение функции: {Z}")
        print()

