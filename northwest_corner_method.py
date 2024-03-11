import numpy as np

def northwest_corner_method(a, b):
    i, j = 0, 0  # Индексы текущего поставщика и потребителя
    m, n = len(a), len(b)  # Количество поставщиков и потребителей
    X = np.zeros((m, n))  # Опорный план

    while i < m and j < n:
        # Переменная x_ij - объем перевозки от i-го поставщика к j-му потребителю
        x_ij = min(a[i], b[j])

        # Заполнение ячейки опорного плана
        X[i][j] = x_ij

        # Обновление остатка предложения и спроса
        a[i] -= x_ij
        b[j] -= x_ij

        # Правила для перехода к следующему поставщику/потребителю
        if a[i] < b[j]:
            i += 1  # Переход к следующему поставщику
        elif a[i] > b[j]:
            j += 1  # Переход к следующему потребителю
        else:  # supply[i] == demand[j]
            # В случае равенства, необходимо исключить одну из строк/столбцов
            # Исключаем строку/столбец так, чтобы не нарушить баланс общего предложения и спроса
            if i < m - 1 and j < n - 1:
                i += 1
                j += 1
            elif i == m - 1:
                j += 1
            elif j == n - 1:
                i += 1

    return X

if __name__ == "__main__":
    a = np.array([100, 250, 200, 300])
    b = np.array([200, 200, 100, 100, 250])
    c = np.array([
        [10, 7, 4, 1, 4],
        [2, 7, 10, 6, 11],
        [8, 5, 3, 2, 2],
        [11, 8, 12, 16, 13]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([200, 175, 225])
    b = np.array([100, 125, 325, 250, 100])
    c = np.array([
        [5, 7, 4, 2, 5],
        [7, 1, 3, 1, 10],
        [2, 3, 6, 8, 7]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([200, 450, 250])
    b = np.array([100, 125, 325, 250, 100])
    c = np.array([
        [5, 8, 7, 10, 3],
        [4, 2, 2, 5, 6],
        [7, 3, 5, 9, 2]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([250, 200, 200])
    b = np.array([120, 130, 100, 160, 110])
    c = np.array([
        [27, 36, 35, 31, 29],
        [22, 23, 26, 32, 35],
        [35, 42, 38, 32, 39]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([350, 330, 270])
    b = np.array([210, 170, 220, 150, 200])
    c = np.array([
        [3, 12, 9, 1, 7],
        [2, 4, 11, 2, 10],
        [7, 14, 12, 5, 8]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([300, 250, 200])
    b = np.array([210, 170, 220, 150, 200])
    c = np.array([
        [4, 8, 13, 2, 7],
        [9, 4, 11, 9, 17],
        [3, 16, 10, 1, 4]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([350, 200, 300])
    b = np.array([170, 140, 200, 195, 145])
    c = np.array([
        [22, 14, 16, 28, 30],
        [19, 17, 26, 36, 36],
        [37, 30, 31, 39, 41]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([200, 250, 200])
    b = np.array([190, 100, 120, 110, 130])
    c = np.array([
        [28, 27, 18, 27, 24],
        [18, 26, 27, 32, 21],
        [27, 33, 23, 31, 34]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([230, 150, 170])
    b = np.array([140, 90, 160, 110, 150])
    c = np.array([
        [40, 19, 25, 26, 35],
        [42, 25, 27, 15, 38],
        [46, 27, 36, 40, 45]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([200, 300, 250])
    b = np.array([210, 150, 120, 135, 135])
    c = np.array([
        [20, 10, 12, 13, 16],
        [25, 19, 20, 14, 10],
        [17, 18, 15, 10, 17]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)

    a = np.array([200, 350, 300])
    b = np.array([270, 130, 190, 150, 110])
    c = np.array([
        [24, 50, 45, 27, 15],
        [20, 32, 40, 35, 30],
        [22, 16, 18, 28, 20]
    ])
    X = northwest_corner_method(a, b)
    Z = np.sum(X * c)
    print(Z)
