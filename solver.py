from flask import Flask, render_template, request

import numpy as np
from simplex_method import run_simplex_analysis
from m_method import run_m_method_analysis

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.form)

        # Определяем количество переменных и уравнений
        variables_count = int(request.form['variablesCount'])
        equations_count = int(request.form['equationsCount'])

        # Определяем коэффициенты целевого уравнения
        objective_coeffs = []
        for i in range(variables_count):
            coeff_name = f"c_{i + 1}"
            if coeff_name in request.form:
                objective_coeffs.append(request.form[coeff_name])

        # Определяем коэффициенты уравнений, отношения и правые части уравнений
        coefficients, relations, right_sides = [], [], []
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
            relation_name = f"relation_{i + 1}"
            if relation_name in request.form:
                relations.append(request.form[relation_name])

        # Преобразуем строковые значения в целочисленные
        objective_coeffs = [int(objective_coeff) for objective_coeff in objective_coeffs]
        coefficients = [[int(coefficient) for coefficient in row] for row in coefficients]
        right_sides = [int(right_side) for right_side in right_sides]

        objective_coeffs_before = np.array(objective_coeffs)
        coefficients_before = np.array(coefficients)

        coefficients = np.array(coefficients)
        for i in range(len(coefficients)):
            column = np.zeros((len(coefficients), 1))
            if relations[i] == '<=':
                column[i] = 1
                coefficients = np.hstack((coefficients, column))
                objective_coeffs.append(0)
            elif relations[i] == '>=':
                column[i] = -1
                coefficients = np.hstack((coefficients, column))
                objective_coeffs.append(0)

        count = 0
        for col in range(coefficients.shape[1]):
            if np.sum(coefficients[:, col]) == 1 and np.count_nonzero(coefficients[:, col] == 1) == 1:
                count += 1

        use_m_method = True if count != len(right_sides) else False

        A = np.array(coefficients)
        b = np.array(right_sides)
        c = np.array(objective_coeffs) * -1

        print(c)
        print(A)
        print(b)

        if not use_m_method:
            run_simplex_analysis(objective_coeffs_before, coefficients_before, b)
            return render_template('simplex_method.html')
        else:
            run_m_method_analysis(c, A, b)
            return render_template('m_method.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
