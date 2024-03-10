from fractions import Fraction

from bs4 import BeautifulSoup
import numpy as np
from numpy._typing import NDArray
import pandas as pd

pd.set_option('display.precision', 2)


def to_fraction(x):
    return str(Fraction(x).limit_denominator())


def print_tableau(tableau, var_names, index_names, iterations):
    df = pd.DataFrame(tableau, columns=var_names, index=index_names)
    df_fraction = df.map(to_fraction)
    df_fraction = df_fraction.to_html(index=False)
    iterations.append(f"<h2>Итерация {len(iterations)}</h2>" + df_fraction)


def simplex_method(c: NDArray, A: NDArray, b: NDArray):
    # Проверка крайних случаев
    if np.any(b < 0):
        print("Вектор \"b\" содержит неотрицательные элементы")
        return [], [], None, []

    # Вспомогательные переменные
    num_vars = len(c)
    num_constraints = len(b)

    # Построение симплекс-таблицы
    c = np.hstack([c * -1, np.zeros(num_constraints + 1)])
    tableau = np.hstack([A, np.eye(num_constraints), b.reshape(-1, 1)])
    tableau = np.vstack([tableau, c])

    vars = ([f"x{i}"
             for i in range(1, num_vars + num_constraints + 1)] + ["b"])

    basis = [
        f"x{i}"
        for i in range(num_constraints + 1, num_constraints + tableau.shape[0])
    ] + ["Z"]

    iterations = []
    while True:
        # Вывод таблицы в начале итерации
        print_tableau(tableau, vars, basis, iterations)

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
        basis[pivot_row] = f"x{pivot_col + 1}"

        # Выбор разрешающего коэффициента
        pivot_element = tableau[pivot_row, pivot_col]

        # Пересчитываем элементы разрешающей строки
        tableau[pivot_row, :] /= pivot_element

        # Пересчитываем элементы остальных строк
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Определение решения
    X = np.zeros(num_vars + num_constraints)
    for i in range(len(b)):
        variable_index = int(basis[i][1:]) - 1
        variable_value = tableau[i, -1]
        X[variable_index] = variable_value

    X_ = np.array(tableau[-1])

    Z = tableau[-1][-1]

    return X, X_, Z, iterations


def create_solution_table(X, Z):
    """
    Создает таблицу с интерпретацией решения.
    """
    columns = [
        "Основные переменные и целевая функция", "Оптимальные значения",
        "Интерпретация"
    ]
    data = {col: [] for col in columns}

    # Добавление значений неизвестных
    for i, value in enumerate(X):
        if value != 0:
            data[columns[0]].append(f"x{i + 1}")
            data[columns[1]].append(value)
            data[columns[2]].append(
                f"Объем производства продукции П{i+1} должен составлять {value:.2f} ед. за исследуемый временной период."
            )

    # Добавление значения целевой функции Z
    data[columns[0]].append("Z")
    data[columns[1]].append(Z)
    data[columns[2]].append(
        f"Доход от реализации продукции составит {Z:.2f} ден.ед. за исследуемый временной период"
    )

    return pd.DataFrame(data)


def create_resource_status_table(X, b):
    """
    Создает таблицу статуса ресурсов.
    """
    columns = [
        "Ресурс", "Ассоциируемая дополнительная переменная",
        "Значение ассоциируемой дополнительной переменной", "Статус ресурса",
        "Интерпретация"
    ]
    data = {col: [] for col in columns}

    for i in range(len(b)):
        data[columns[0]].append(f"P{i+1}")
        data[columns[1]].append(f"x{i + len(b) + 1}")
        data[columns[2]].append(X[i + len(b)])
        status = "Дефицитный" if X[i + len(b)] == 0 else "Недефицитный"
        data[columns[3]].append(status)
        if status == "Дефицитный":
            data[columns[4]].append(
                f"Ресурс Р{i+1} полностью потреблен. Увеличение запаса ресурса Р{i+1} позволит увеличить доход от реализации продукции"
            )
        else:
            data[columns[4]].append(
                f"Ресурс Р{i+1} потреблен не полностью. Увеличение запаса ресурса Р{i+1} не позволит увеличить доход от реализации продукции"
            )

    return pd.DataFrame(data)


def create_resource_value_analysis_table(X_, b):
    """
    Создает таблицу анализа ценности ресурса.
    """
    columns = [
        "Дополнительная переменная", "Оценка дополнительной переменной",
        "Интерпретация"
    ]
    data = {col: [] for col in columns}

    aux_var_scores = X_[len(b):-1]
    sorted_indices = sorted(range(len(aux_var_scores)),
                            key=lambda i: aux_var_scores[i],
                            reverse=True)

    for i, index in enumerate(sorted_indices):
        data[columns[0]].append(f"x{len(b) + index + 1}")
        data[columns[1]].append(aux_var_scores[index])
        if i == 0:
            data[columns[2]].append(
                f"Ресурс Р{index + 1} имеет наибольшую ценность.")
        else:
            data[columns[2]].append("")

    # Объединение интерпретаций в одну строку для первого элемента
    interpretation = f"Порядок увеличения ценности: " + ", ".join(
        [f"Р{index + 1}" for index in sorted_indices])
    if data[columns[0]]:
        data[columns[2]][0] = interpretation

    return pd.DataFrame(data)


def adjust_resource_value_table(html_table):
    """
    Делает корректировки в HTML таблице анализа ценности ресурса.
    """
    soup = BeautifulSoup(html_table, 'html.parser')
    rows = soup.find_all('tr')[1:]
    for i, row in enumerate(rows):
        tds = row.find_all('td')
        if i == 0:
            tds[2]['rowspan'] = str(len(rows))
        else:
            if len(tds) > 2:
                tds[2].decompose()

    return str(soup)


def write_html_document(method_iterations, solution_table,
                        resource_status_table, resource_value_table, filename: str):
    """
    Записывает результат в HTML документ
    """
    html_document = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>Результаты симплекс-метода</title>
    <style>
        table, th, td {{
            border: 1px solid black;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text-align: center;
        }}
    </style>
    </head>
    <body>
    {method_iterations}
    <h2>Интерпретация оптимального решения</h2>
    {solution_table}
    <h2>Анализ статуса ресурсов</h2>
    {resource_status_table}
    <h2>Анализ ценности ресурса</h2>
    {resource_value_table}
    </body>
    </html>
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(html_document)


def run_simplex_analysis(c, A_ub, b_ub):
    # Решение тестового примера методом симплекса
    X, X_, Z, iterations = simplex_method(c, A_ub, b_ub)

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
        X, b_ub).to_html(index=False)

    # Создание и корректировка таблицы ценности ресурса для HTML документа
    resource_value_table = create_resource_value_analysis_table(
        X_, b_ub).to_html(index=False)
    resource_value_table = adjust_resource_value_table(resource_value_table)

    # Сохранение результатов в HTML документ
    write_html_document(method_iterations, solution_table,
                        resource_status_table, resource_value_table, 'templates/simplex_method.html')


if __name__ == "__main__":
    # Важно привести систему к канонической форме

    # Тестовые примеры
    # c = np.array([1, -1, -3])
    # A_ub = np.array([[2, -1, 1], [-4, 2, -1], [3, 0, 1]])
    # b_ub = np.array([1, 2, 5])

    # """
    #     7x_1 + 5x_2 = Z
    #     
    #     2x_1 + 3x_2 = 90
    #     3x_1 + 2x_2 = 120
    # """
    # c = np.array([-7, -5])
    # A_ub = np.array([[2, 3], [3, 2]])
    # b_ub = np.array([90, 120])
    #
    # c = np.array([-6, -8, -5])
    # A_ub = np.array([[4, 1, 1], [2, 2, 1], [4, 2, 1]])
    # b_ub = np.array([1800, 2000, 3200])

    # c = np.array([-3, -1, -3])
    # A_ub = np.array([[2, 1, 1], [1, 2, 3], [2, 2, 1]])
    # b_ub = np.array([2, 5, 6])

    # run_simplex_analysis(c, A_ub, b_ub)
    pass
