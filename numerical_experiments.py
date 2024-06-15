import matplotlib.pyplot as plt
from mprod import generate_haar, generate_dct, m_prod
import numpy as np
from algo import StarAlgebra as star
import experimnt_utils as eu

formula_norm_res = '$\Vert A\cdot\overline{X}-\overline{B}\Vert_F/\Vert \overline{B}\Vert_F$'
formula_ATR = '$\Vert A^T\cdot\overline{R}\Vert_F/\Vert A\Vert_F\Vert \overline{R}\Vert_F$'
formula = formula_ATR


def get_ATR(list_iter_X, tenA, omatB, funM):
    tenA_hat = funM(tenA)
    omatR_hats = [star.facewise_mult(tenA_hat, X) - funM(omatB) for X in list_iter_X]
    iterative_ATR = [star.Fnorm(star.facewise_mult(tenA_hat.transpose((1, 0, 2)), R))/star.Fnorm(R) for R in omatR_hats]
    iterative_residual = [r / star.Fnorm(tenA_hat) for r in iterative_ATR]
    return iterative_residual


def scalar_normalization(dct):
    tenA, omatB, funM, invM = eu.randn_tenA_omatB_omatX_M(dct=dct)
    star_lsqr = star(funM, invM)
    _ = star_lsqr.fit_LSQR_predict(tenA, omatB)
    iterative_residual = get_ATR(star_lsqr.iterative_solutions, tenA, omatB, funM)
    dict_of_lines = {'tubal normalization': iterative_residual}

    _ = star_lsqr.fit_LSQR_scalar_predict(tenA, omatB)
    iterative_residual = get_ATR(star_lsqr.iterative_solutions, tenA, omatB, funM)
    dict_of_lines['scalar normalization'] = iterative_residual

    return tenA.shape, dict_of_lines


def scalar_normalization_rescaled_tensor(dct):
    tenA, omatB, funM, invM = eu.randn_not_scaled_tenA_omatB_omatX_M(dct=dct)
    star_lsqr = star(funM, invM)
    _ = star_lsqr.fit_LSQR_predict(tenA, omatB)
    iterative_residual = get_ATR(star_lsqr.iterative_solutions, tenA, omatB, funM)
    dict_of_lines = {'tubal normalization': iterative_residual}

    _ = star_lsqr.fit_LSQR_scalar_predict(tenA, omatB)
    iterative_residual = get_ATR(star_lsqr.iterative_solutions, tenA, omatB, funM)
    dict_of_lines['scalar normalization'] = iterative_residual

    return tenA.shape, dict_of_lines


Asize, dict_of_lines = scalar_normalization(dct=True)
eu.plot_iterative(
    dict_of_lines,
    'comparison of LSQR convergence for different normalization',
    'scalar_lsqr',
    'iterations',
    formula,
    'dct',
    Asize)

Asize, dict_of_lines = scalar_normalization_rescaled_tensor(dct=True)
eu.plot_iterative(
    dict_of_lines,
    'comparison of LSQR convergence for different normalization\nfrontal slices of different scale (up to $10^6$)',
    'scalar_lsqr_rescaled',
    'iterations',
    formula,
    'dct',
    Asize)



def coherency_tensor(cond_number, height, width, depth, horizontal_mix=False):
    tenA, omatB, funM, invM = eu.generate_coh_cond_system('coherent', cond_number, horizontal_mix, height, width, depth)
    star_lsqr = star(funM, invM)
    _ = star_lsqr.fit_blendenpik_predict(tenA, omatB)
    iterative_residual = get_ATR(star_lsqr.iterative_solutions, tenA, omatB, funM)
    dict_of_lines = {'coherent': iterative_residual}

    # tenA, omatB, funM, invM = eu.generate_coh_cond_system('semi-coherent', cond_number, horizontal_mix, height, width, depth)
    # star_lsqr = star(funM, invM)
    # _ = star_lsqr.fit_blendenpik_predict(tenA, omatB)
    # iterative_residual = get_ATR(star_lsqr.iterative_solutions, tenA, omatB, funM)
    # dict_of_lines['semi-coherent'] = iterative_residual

    tenA, omatB, funM, invM = eu.generate_coh_cond_system('incoherent', cond_number, horizontal_mix, height, width, depth)
    star_lsqr = star(funM, invM)
    _ = star_lsqr.fit_blendenpik_predict(tenA, omatB)
    iterative_residual = get_ATR(star_lsqr.iterative_solutions, tenA, omatB, funM)
    dict_of_lines['incoherent'] = iterative_residual


    return tenA.shape, dict_of_lines


Asize, dict_of_lines = coherency_tensor(10, 1000, 20, 10, horizontal_mix=False)
eu.plot_iterative(
    dict_of_lines,
    'comparison of LSQR convergence for different coherency\nwell conditioned tensor',
    'coherency_convergence_well_cond_not_mixed',
    'iterations',
    formula,
    'random orthogonal',
    Asize)


Asize, dict_of_lines = coherency_tensor(10, 1000, 20, 10, horizontal_mix=True)
eu.plot_iterative(
    dict_of_lines,
    'comparison of LSQR convergence for different coherency\nwell conditioned tensor',
    'coherency_convergence_well_cond_mixed',
    'iterations',
    formula,
    'random orthogonal',
    Asize)

Asize, dict_of_lines = coherency_tensor(10**6, 1000, 25, 10, horizontal_mix=False)
eu.plot_iterative(
    dict_of_lines,
    'comparison of LSQR convergence for different coherency\nill conditioned tensor',
    'coherency_convergence_ill_cond_not_mixed',
    'iterations',
    formula,
    'random orthogonal',
    Asize)


Asize, dict_of_lines = coherency_tensor(10**6, 1000, 25, 10, horizontal_mix=True)
eu.plot_iterative(
    dict_of_lines,
    'comparison of LSQR convergence for different coherency\nill conditioned tensor',
    'coherency_convergence_ill_cond_mixed',
    'iterations',
    formula,
    'random orthogonal',
    Asize)