import numpy as np
from temporal_disagg.conversion_matrix import CovMatrixBuilder

def test_power_matrix_calculation():
    builder = CovMatrixBuilder()
    pm = builder.power_matrix_calculation(3)
    
    expected = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    assert np.array_equal(pm, expected)

def test_q_calculation():
    builder = CovMatrixBuilder()
    pm = builder.power_matrix_calculation(3)
    q_matrix = builder.q_calculation(rho=0.5, pm=pm)

    assert q_matrix.shape == (3, 3)  # La matriz debe ser cuadrada
    assert np.all(q_matrix >= 0)  # No debe haber valores negativos
