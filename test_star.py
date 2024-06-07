import numpy as np
import pytest
from algo import StarAlgebra as star
from mprod import generate_haar, generate_dct, m_prod


@pytest.fixture
def tenA_omatB_omatX_M():
    m, p, n = 100, 20, 10
    A_rec = np.random.randn(m, p, n)
    tenA_tr = star.facewise_mult(A_rec.transpose((1, 0, 2)), A_rec)
    omatX_tr = np.random.randn(p, 1, n)
    omatB_tr = star.facewise_mult(tenA_tr, omatX_tr)

    funM, invM = generate_haar(n, 21)
    tenA = invM(tenA_tr)
    omatX = invM(omatX_tr)
    omatB = invM(omatB_tr)
    return tenA, omatX, omatB, funM, invM


def test_normalization(tenA_omatB_omatX_M):
    tenA, omatX, omatB, funM, invM = tenA_omatB_omatX_M
    star_cg = star(funM, invM)
    omatB_norm, beta_norm = star_cg.normalize(omatB)
    norm_B_norm = star.Mnorm(omatB_norm)
    assert np.isclose(norm_B_norm, 1).all()

    omatB_norm, beta_norm = star_cg.normalize(omatB, funM, invM)
    norm_B_tr_norm = star.Mnorm(omatB_norm, funM)
    assert np.isclose(norm_B_tr_norm, 1).all()


def test_basic(tenA_omatB_omatX_M):
    tenA, omatX, omatB, funM, invM = tenA_omatB_omatX_M
    star_cg = star(funM, invM)
    omatX_pred = star_cg.fitCG_predict(tenA, omatB)
    print(star.Fnorm(omatX_pred-omatX))
    assert star.Fnorm(omatX_pred-omatX) < 10**-6



