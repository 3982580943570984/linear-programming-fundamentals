from scipy.optimize import linprog

# Коэффициенты функции цели (для минимизации, умножаем на -1)
c = [1, -1, -3]

# Коэффициенты левых частей ограничений (A_ub * x <= b_ub)
A_ub = [[2, -1, 1], [-4, 2, -1], [3, 0, 1]]

# Правые части ограничений
b_ub = [1, 2, 5]

# Границы переменных (все переменные >= 0)
bounds = [(0, None), (0, None), (0, None)]

# Решение задачи линейного программирования с помощью симплекс-метода
result = linprog(c,
                 A_ub=A_ub,
                 b_ub=b_ub,
                 bounds=bounds,
                 method='revised simplex',
                 options={
                     'disp': True,
                 })

# print(result.x)
