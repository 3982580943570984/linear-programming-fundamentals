import numpy as np


def count_identity_columns(matrix):
    """
    This function counts the number of identity columns in a given matrix.
    
    Parameters:
    - matrix (np.array): A numpy array representing the matrix to check.
    
    Returns:
    - int: The number of identity columns in the matrix.
    """
    n_rows, n_cols = matrix.shape
    identity_column_count = 0

    for col in range(n_cols):
        # Extract the column
        current_col = matrix[:, col]

        # Count the number of ones and zeroes in the column
        ones_count = np.count_nonzero(current_col == 1)
        zeroes_count = np.count_nonzero(current_col == 0)

        # Check if the column is an identity column
        if (ones_count == 1) and (zeroes_count == n_rows - 1):
            identity_column_count += 1

    return identity_column_count


def rearrange_to_identity(matrix):
    """
    This function attempts to rearrange the rows of a given matrix to form an identity matrix,
    if possible, when the number of identity columns is equal to the number of rows.
    
    Parameters:
    - matrix (np.array): A numpy array representing the matrix to check and rearrange.
    
    Returns:
    - np.array: The rearranged matrix if it can form an identity matrix, otherwise the original matrix.
    - bool: True if the matrix can be rearranged to form an identity matrix, False otherwise.
    """
    n_rows, n_cols = matrix.shape

    # First, check if the number of identity columns equals the number of rows
    identity_column_count = count_identity_columns(matrix)
    if identity_column_count != n_rows:
        return matrix, False  # Cannot form an identity matrix

    # Attempt to rearrange rows to form an identity matrix
    rearranged_matrix = np.zeros_like(matrix)
    for col in range(n_cols):
        current_col = matrix[:, col]
        if np.count_nonzero(current_col == 1) == 1 and np.count_nonzero(
                current_col == 0) == n_rows - 1:
            # Find the row index where the 1 is located in the current column
            row_index = np.where(current_col == 1)[0][0]
            rearranged_matrix[row_index, col] = 1

    # Verify if the rearranged matrix is indeed an identity matrix
    is_identity = np.array_equal(rearranged_matrix, np.eye(n_rows))

    return rearranged_matrix, is_identity


matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

print(matrix, "\n")

matrix, result = rearrange_to_identity(matrix)

print(matrix, result)
