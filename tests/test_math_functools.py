import numpy as np
import math

from thesis.utils import math_functools

def test_l2_and_inner_products():
    # 3^2 + 4^ = 25, sqrt(25) = 5
    v = np.array([3.0, 4.0])
    assert math.isclose(math_functools.l2_norm(v), 5.0)
    # 5^2 = 25, sqrt(25) = 5
    assert np.allclose(math_functools.batch_l2_norm(np.array([[3.0,4.0],[0.0,5.0]])), np.array([5.0,5.0]))
    # (3 * 1) + (2 * 4) = (3 + 8) = 11
    assert math.isclose(math_functools.inner_product(v, np.array([1.0, 2.0])), 11.0)
    # (1 * 3) + (2 * 4) = 11
    # (1 * 5) + (2 * 6) = 17
    assert np.allclose(math_functools.batch_inner_product(np.array([1.0,2.0]), np.array([[3.0,4.0],[5.0,6.0]])), np.array([11.0, 17.0]))

def test_cosine_similarity():
    u = np.array([1.0, 0.0])
    vs = np.array([[1.0,0.0],[0.0,1.0]])
    cos_sim_result = math_functools.batch_cosine_similarity(u, vs)
    assert np.allclose(cos_sim_result, np.array([1.0, 0.0]))

# TODO: implement test for poincare distance
def test_poincare_distance():
    assert 1 == 1
