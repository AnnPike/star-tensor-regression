import numpy as np
import pytest
from algo import StarAlgebra as star
from mprod import generate_haar


@pytest.fixture
def tenA_sq_omatB_omatX_M():
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


@pytest.fixture
def tenA_omatB_omatX_M():
    m, p, n = 100, 20, 10
    A_rec_tr = np.random.randn(m, p, n)
    omatX_tr = np.random.randn(p, 1, n)
    omatB_tr = star.facewise_mult(A_rec_tr, omatX_tr)

    funM, invM = generate_haar(n, 21)
    tenA = invM(A_rec_tr)
    omatX = invM(omatX_tr)
    omatB = invM(omatB_tr)
    return tenA, omatX, omatB, funM, invM


def test_normalization(tenA_sq_omatB_omatX_M):
    tenA, omatX, omatB, funM, invM = tenA_sq_omatB_omatX_M
    star_cg = star(funM, invM)
    omatB_norm, beta_norm = star_cg.normalize(omatB)
    norm_B_norm = star.Mnorm(omatB_norm)
    assert ((beta_norm*omatB_norm-omatB) < 10**-9).all()
    assert np.isclose(norm_B_norm, 1).all()

    omatB_norm, beta_norm = star_cg.normalize(omatB, funM, invM)
    norm_B_tr_norm = star.Mnorm(omatB_norm, funM)
    assert (invM(funM(beta_norm)*funM(omatB_norm))-omatB < 10**-9).all()
    assert np.isclose(norm_B_tr_norm, 1).all()


def test_CG(tenA_sq_omatB_omatX_M):
    tenA, omatX, omatB, funM, invM = tenA_sq_omatB_omatX_M
    star_cg = star(funM, invM)
    omatX_pred = star_cg.fitCG_predict(tenA, omatB)
    print(star.Fnorm(omatX_pred-omatX))
    assert star.Fnorm(omatX_pred-omatX) < 10**-6

    iterative_norm_tr = [star.Fnorm(invM(X) - omatX) for X in star_cg.iterative_solutions_CG]
    assert (np.array(iterative_norm_tr[:-1]) >= np.array(iterative_norm_tr[1:])).all()


def test_LSQR(tenA_omatB_omatX_M):
    tenA, omatX, omatB, funM, invM = tenA_omatB_omatX_M
    star_lsqr = star(funM, invM)
    omatX_pred = star_lsqr.fit_LSQR_predict(tenA, omatB)
    print(star.Fnorm(omatX_pred-omatX))
    assert star.Fnorm(omatX_pred-omatX) < 10**-6

    iterative_norm_tr = [star.Fnorm(invM(X) - omatX) for X in star_lsqr.iterative_solutions_LSQR]
    assert (np.array(iterative_norm_tr[:-1]) >= np.array(iterative_norm_tr[1:])).all()




