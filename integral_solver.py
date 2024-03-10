from flask import Flask, render_template, request
from sympy import Matrix

import numpy as np

import globals as g
from utilities import to_canonical
from cutting_plane_method import *
from dual_simplex_method import *

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Определяем количество переменных и уравнений
        variables_count = int(request.form['variables_count'])
        equations_count = int(request.form['equations_count'])

        # Определяем коэффициенты целевого уравнения
        objective_coeffs = []
        for i in range(variables_count):
            coeff_name = f"c_{i + 1}"
            if coeff_name in request.form:
                objective_coeffs.append(request.form[coeff_name])

        # Определяем коэффициенты уравнений, отношения и правые части уравнений
        coefficients, right_sides, signs = [], [], []
        for i in range(equations_count):
            # Определяем коэффициенты уравнений
            equation_coeffs = []
            for j in range(variables_count):
                coeff_name = f"A_{i + 1}_{j + 1}"
                if coeff_name in request.form:
                    equation_coeffs.append(request.form[coeff_name])
            coefficients.append(equation_coeffs)

            # Определяем отношения
            right_side_name = f"b_{i + 1}"
            if right_side_name in request.form:
                right_sides.append(request.form[right_side_name])

            # Определяем правые части уравнений
            sign_name = f"sign_{i + 1}"
            if sign_name in request.form:
                signs.append(request.form[sign_name])

        # Преобразуем строковые значения в целочисленные
        objective_coeffs = [int(objective_coeff) for objective_coeff in objective_coeffs]
        coefficients = [[int(coefficient) for coefficient in row] for row in coefficients]
        right_sides = [int(right_side) for right_side in right_sides]

        A = np.array(coefficients).astype(float)
        b = np.array(right_sides).astype(float)
        c = np.array(objective_coeffs).astype(float)

        A, b, c = to_canonical(A, b, c, signs)

        g.reset_globals()

        x = cutting_plane_method(A.copy(), b.copy(), c.copy())

        Z = -sum(ci * xi for ci, xi in zip(c, x))

        create_interpretation(x, Z)

        return render_template('interpretation.html')
    else:
        return render_template('index.html')


def create_optimal_solution(X: np.ndarray) -> str:
    return f"$$X = {latex(X.reshape(1, -1))}$$"


def create_solution_extremum(Z: float) -> str:
    return f"$$Z = {Z}$$"


def create_interpretation(X: np.ndarray, Z: float):
    m_method_interpretations = create_m_method_interpretations()
    cutting_plane_method_interpretations = create_cutting_plane_interpretations()
    dual_simplex_method_interpretations = create_dual_simplex_interpretations()
    simplex_interpretations = create_simplex_interpretations()

    html_document = f"""
    <!doctype html>
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
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
    <h1>Оптимальное решение</h1>
    {create_optimal_solution(X)}
    <h1>Экстремум целевой функции</h1>
    {create_solution_extremum(Z)}
    <h1>М-метод</h1>
    {"".join(m_method_interpretations) if len(m_method_interpretations) != 0 else "<h2>М-метод не был применен</h2>"}
    <h1>Метод Гомори</h1>
    {"".join([interpretation + equation for interpretation, equation in zip(cutting_plane_method_interpretations, g.cutting_plane_method_equations)])}
    <h1>Двойственный симплекс-метод</h1>
    {"".join(dual_simplex_method_interpretations)}
    <h1>Симплекс-метод</h1>
    {"".join(simplex_interpretations)}
    </body>
    </html>
    """
    with open('templates/interpretation.html', 'w', encoding='utf-8') as file:
        file.write(html_document)


if __name__ == "__main__":
    app.run(debug=True)
