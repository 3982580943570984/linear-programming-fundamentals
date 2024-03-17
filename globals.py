from collections import defaultdict
from typing import List, Tuple, DefaultDict

import numpy as np

m_method_iterations: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
cutting_plane_method_iterations: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
cutting_plane_method_equations: List[str] = []
interpretations: DefaultDict[int, List[str]] = defaultdict(list)
dual_simplex_method_iterations: DefaultDict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
simplex_method_iterations: DefaultDict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)

def reset_globals():
    m_method_iterations.clear()
    cutting_plane_method_iterations.clear()
    cutting_plane_method_equations.clear()
    interpretations.clear()
    dual_simplex_method_iterations.clear()
    simplex_method_iterations.clear()
