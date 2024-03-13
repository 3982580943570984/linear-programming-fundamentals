import numpy as np


def balance_supply_demand(supply: np.ndarray, demand: np.ndarray):
    supply_total = np.sum(supply)
    demand_total = np.sum(demand)
    
    if supply_total != demand_total:
        if supply_total > demand_total:
            demand = np.append(demand, supply_total - demand_total)
        else:
            supply = np.append(supply, demand_total - supply_total)

    return supply, demand


def northwest_corner_method(supply: np.ndarray, demand: np.ndarray):
    i, j = 0, 0
    m, n = len(supply), len(demand)
    X = np.zeros((m, n))
    while i != m and j != n:
        if supply[i] <= demand[j]:
            demand[j] -= supply[i]
            X[i][j] = supply[i]
            i += 1
        else:
            supply[i] -= demand[j]
            X[i][j] = demand[j]
            j += 1
    return X


if __name__ == "__main__":
    supply = np.array([100, 250, 200, 300])
    demand = np.array([200, 200, 100, 100, 250])
    cost = np.array([
        [10, 7, 4, 1, 4],
        [2, 7, 10, 6, 11],
        [8, 5, 3, 2, 2],
        [11, 8, 12, 16, 13]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply.copy(), demand.copy())
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([200, 175, 225])
    demand = np.array([100, 125, 325, 250, 100])
    cost = np.array([
        [5, 7, 4, 2, 5],
        [7, 1, 3, 1, 10],
        [2, 3, 6, 8, 7]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([200, 450, 250])
    demand = np.array([100, 125, 325, 250, 100])
    cost = np.array([
        [5, 8, 7, 10, 3],
        [4, 2, 2, 5, 6],
        [7, 3, 5, 9, 2]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([250, 200, 200])
    demand = np.array([120, 130, 100, 160, 110])
    cost = np.array([
        [27, 36, 35, 31, 29],
        [22, 23, 26, 32, 35],
        [35, 42, 38, 32, 39]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([350, 330, 270])
    demand = np.array([210, 170, 220, 150, 200])
    cost = np.array([
        [3, 12, 9, 1, 7],
        [2, 4, 11, 2, 10],
        [7, 14, 12, 5, 8]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([300, 250, 200])
    demand = np.array([210, 170, 220, 150, 200])
    cost = np.array([
        [4, 8, 13, 2, 7],
        [9, 4, 11, 9, 17],
        [3, 16, 10, 1, 4]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([350, 200, 300])
    demand = np.array([170, 140, 200, 195, 145])
    cost = np.array([
        [22, 14, 16, 28, 30],
        [19, 17, 26, 36, 36],
        [37, 30, 31, 39, 41]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([200, 250, 200])
    demand = np.array([190, 100, 120, 110, 130])
    cost = np.array([
        [28, 27, 18, 27, 24],
        [18, 26, 27, 32, 21],
        [27, 33, 23, 31, 34]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([230, 150, 170])
    demand = np.array([140, 90, 160, 110, 150])
    cost = np.array([
        [40, 19, 25, 26, 35],
        [42, 25, 27, 15, 38],
        [46, 27, 36, 40, 45]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([200, 300, 250])
    demand = np.array([210, 150, 120, 135, 135])
    cost = np.array([
        [20, 10, 12, 13, 16],
        [25, 19, 20, 14, 10],
        [17, 18, 15, 10, 17]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)

    supply = np.array([200, 350, 300])
    demand = np.array([270, 130, 190, 150, 110])
    cost = np.array([
        [24, 50, 45, 27, 15],
        [20, 32, 40, 35, 30],
        [22, 16, 18, 28, 20]
    ])
    supply, demand = balance_supply_demand(supply, demand)
    X = northwest_corner_method(supply, demand)
    Z = np.sum(X * cost)
    print(Z)
