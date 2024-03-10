from typing import List, Tuple

import numpy as np

m_method_iterations: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
cutting_plane_method_iterations: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
cutting_plane_method_equations: List[str] = []
dual_simplex_method_iterations: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
simplex_method_iterations: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

def reset_globals():
    m_method_iterations.clear()
    cutting_plane_method_iterations.clear()
    cutting_plane_method_equations.clear()
    dual_simplex_method_iterations.clear()
    simplex_method_iterations.clear()
