from typing import List, Tuple, Any

import numpy as np
from numpy.typing import NDArray


def to_canonical(A: NDArray[Any], b: NDArray[Any], c: NDArray[Any], signs: List[str]) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    slack_size = sum(sign in ['<=', '<', '>=', '>'] for sign in signs)

    for i, sign in enumerate(signs):
        if sign == '>=' or sign == '>':
            b[i] *= -1
            A[i] *= -1
    c *= -1

    if slack_size != 0:
        identity_matrix = np.eye(slack_size)
        if slack_size != A.shape[0]:
            identity_matrix = np.vstack([identity_matrix, np.zeros(A.shape[0] - slack_size)])

        A = np.hstack([A, identity_matrix])
        c = np.hstack([c, np.zeros(slack_size)])

    return A, b, c


def selective_rounding(numbers, tolerance=1e-9):
    rounded_numbers = np.empty_like(numbers)
    for i, num in enumerate(numbers):
        nearest_int = np.rint(num)
        if np.abs(num - nearest_int) <= tolerance:
            rounded_numbers[i] = nearest_int
        else:
            rounded_numbers[i] = num
    return rounded_numbers
