import numpy as np

max_int, n = 100, 5
initial_cost_matrix = np.random.randint(max_int, size=(n, n), dtype=int)
cost_matrix = initial_cost_matrix.copy()
mask_matrix = np.zeros((n, n), dtype=int)
row_cover = np.zeros(n, dtype=int)
col_cover = np.zeros(n, dtype=int)
path_row_0, path_col_0 = 0, 0


def kuhn_munkres():
    done = False
    step = 1
    while not done:
        # print(f"Cost matrix:\n{cost_matrix}")
        # print(f"Mask matrix:\n{mask_matrix}")

        if step == 1:
            step = step_one()
        elif step == 2:
            step = step_two()
        elif step == 3:
            step = step_three()
        elif step == 4:
            step = step_four()
        elif step == 5:
            step = step_five()
        elif step == 6:
            step = step_six()
        elif step == 7:
            done = True


def step_one():
    for i in range(cost_matrix.shape[0]):
        min_value = np.min(cost_matrix[i])
        cost_matrix[i] -= min_value
    return 2


def step_two():
    for row in range(n):
        for col in range(n):
            if cost_matrix[row, col] == 0 and row_cover[row] == 0 and col_cover[col] == 0:
                mask_matrix[row, col] = 1
                row_cover[row] = 1
                col_cover[col] = 1

    row_cover.fill(0)
    col_cover.fill(0)

    return 3


def step_three():
    for row in range(n):
        for col in range(n):
            if mask_matrix[row, col] == 1:
                col_cover[col] = 1

    col_count = col_cover.sum()

    if col_count >= n:
        return 7
    else:
        return 4


def find_a_zero():
    for i in range(n):
        for j in range(n):
            if cost_matrix[i, j] == 0 and row_cover[i] == 0 and col_cover[j] == 0:
                return i, j
    return -1, -1


def find_star_in_row(row):
    for j in range(n):
        if mask_matrix[row, j] == 1:
            return j
    return -1


def step_four():
    done = False
    while not done:
        row, col = find_a_zero()
        if row == -1:
            done = True
            return 6

        mask_matrix[row, col] = 2

        star_col = find_star_in_row(row)
        if star_col == -1:
            done = True
            global path_row_0, path_col_0
            path_row_0, path_col_0 = row, col
            return 5

        row_cover[row] = 1
        col_cover[star_col] = 0

def find_star_in_col(col):
    for i in range(n):
        if mask_matrix[i, col] == 1:
            return i
    return -1

def find_prime_in_row(row):
    for j in range(n):
        if mask_matrix[row, j] == 2:
            return j
    return -1

def augment_path(path, path_count):
    for p in range(path_count):
        if mask_matrix[path[p][0], path[p][1]] == 1:
            mask_matrix[path[p][0], path[p][1]] = 0
        else:
            mask_matrix[path[p][0], path[p][1]] = 1

def clear_covers():
    row_cover.fill(0)
    col_cover.fill(0)

def erase_primes():
    mask_matrix[mask_matrix == 2] = 0

def step_five():
    path_count = 1
    path = [(path_row_0, path_col_0)]
    
    done = False
    while not done:
        r = find_star_in_col(path[-1][1])
        if r != -1:
            path.append((r, path[-1][1]))
            path_count += 1
            c = find_prime_in_row(r)
            path.append((r, c))
            path_count += 1
        else:
            done = True

    augment_path(path, path_count)
    clear_covers()
    erase_primes()

    return 3


def find_smallest():
    minval = np.max(cost_matrix) + 1
    for r in range(n):
        for c in range(n):
            if row_cover[r] == 0 and col_cover[c] == 0:
                if cost_matrix[r, c] < minval:
                    minval = cost_matrix[r, c]
    return minval


def step_six():
    minval = find_smallest()

    for r in range(n):
        if row_cover[r] == 1:
            cost_matrix[r] += minval

    for c in range(n):
        if col_cover[c] == 0:
            cost_matrix[:, c] -= minval

    return 4


if __name__ == "__main__":
    k = 0
    while True:
        print("#{} -----------------------------------------".format(k))
        k += 1

        n = 50
        initial_cost_matrix = np.random.randint(max_int, size=(n, n), dtype=int)
        cost_matrix = initial_cost_matrix.copy()
        mask_matrix = np.zeros((n, n), dtype=int)
        row_cover = np.zeros(n, dtype=int)
        col_cover = np.zeros(n, dtype=int)
        path_row_0, path_col_0 = 0, 0

        kuhn_munkres()

        total_cost = 0
        for i in range(n):
            for j in range(n):
                if mask_matrix[i, j] == 1:
                    total_cost += initial_cost_matrix[i, j]

        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(initial_cost_matrix.copy())
        assigned_sum = initial_cost_matrix[rows, cols].sum()

        # Проверка на соответствие рассчитанной суммы
        if total_cost != assigned_sum:
            raise ValueError(
                "#{}: The calculated sums do not match: my answer -> {} != lib answer -> {}"
                .format(k, total_cost, assigned_sum))
        else:
            print("#{}: my answer -> {} == lib answer -> {}".format(k, total_cost, assigned_sum))



