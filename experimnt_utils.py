import matplotlib.pyplot as plt
from mprod import generate_haar, generate_dct, m_prod
import numpy as np
from algo import StarAlgebra as star
from scipy.stats import ortho_group


def randn_tenA_omatB_omatX_M(dct, m=1000, p=20, n=10):
    A_rec_tr = np.random.randn(m, p, n)
    omatX_tr = np.random.randn(p, 1, n)
    omatB_tr = star.facewise_mult(A_rec_tr, omatX_tr)
    if dct:
        funM, invM = generate_dct(n)
    else:
        funM, invM = generate_haar(n, 21)
    tenA = invM(A_rec_tr)
    omatX = invM(omatX_tr)
    omatB = invM(omatB_tr)
    return tenA, omatX, omatB, funM, invM


def randn_tenA_omatB_omatX_M(dct, m=1000, p=20, n=10):
    A_rec_tr = np.random.randn(m, p, n)
    omatB_tr = np.random.randn(m, 1, n)
    if dct:
        funM, invM = generate_dct(n)
    else:
        funM, invM = generate_haar(n, 21)
    tenA = invM(A_rec_tr)
    omatB = invM(omatB_tr)
    return tenA, omatB, funM, invM


def randn_not_scaled_tenA_omatB_omatX_M(dct, m=1000, p=20, n=10):
    A_rec_tr = np.random.randn(m, p, n)
    rescale_tube = np.random.randint(1, 10**6, (1, 1, n))
    A_rec_tr = A_rec_tr*rescale_tube
    omatX_tr = np.random.randn(p, 1, n)
    omatB_tr = star.facewise_mult(A_rec_tr, omatX_tr)
    if dct:
        funM, invM = generate_dct(n)
    else:
        funM, invM = generate_haar(n, 21)
    tenA = invM(A_rec_tr)
    omatX = invM(omatX_tr)
    omatB = invM(omatB_tr)
    return tenA, omatX, omatB, funM, invM


def randn_not_scaled_tenA_omatB_omatX_M(dct, m=1000, p=20, n=10):
    A_rec_tr = np.random.randn(m, p, n)
    rescale_tube = np.random.randint(1, 10**6, (1, 1, n))
    A_rec_tr = A_rec_tr*rescale_tube
    omatB_tr = np.random.randn(m, 1, n)
    if dct:
        funM, invM = generate_dct(n)
    else:
        funM, invM = generate_haar(n, 21)
    tenA = invM(A_rec_tr)
    omatB = invM(omatB_tr)
    return tenA, omatB, funM, invM


LIST_OF_COLORS = ['orange', 'blue', 'green']
LIST_OF_MARKERS = ['s', '^', 'o', 'x']
def plot_iterative(dict_of_lines, title, save_name, xlabel, ylabel, Mname, Asize):
    show_ticks = 5
    fig = plt.figure(figsize=(9, 6))
    for i, d in enumerate(dict_of_lines.items()):
        k, v = d
        plt.plot(v, c=LIST_OF_COLORS[i], label=k, lw=3)
        plt.plot(v, LIST_OF_MARKERS[i], c=LIST_OF_COLORS[i], markevery=5, markersize=15, lw=3, mfc='none')

    plt.legend(fontsize=17)
    plt.xlabel(xlabel, fontsize=17)
    plt.xticks(range(0, len(v), 5), fontsize=15)
    plt.yscale("log")
    plt.ylabel(ylabel, fontsize=17)
    plt.yticks(fontsize=15)
    plt.grid(visible=True, which='major', axis='both', ls=':', alpha=0.5)

    fig.suptitle(f'{title}\n height = {Asize[0]}, width = {Asize[1]}, depth = {Asize[2]}, invertible transform = {Mname}', fontsize=17)
    fig.tight_layout()

    fig.savefig('plots/' + save_name)
    plt.close()


def generate_tall_matrix(height: int, width: int, cond_number: float, eigenmin: float = 1):
    H = ortho_group.rvs(dim=height)
    low = eigenmin
    high = eigenmin*cond_number
    eigen = np.linspace(low, high, width)
    LAM = np.zeros((width, width))
    np.fill_diagonal(LAM, eigen)
    P = np.matmul(H[:, :width], LAM)
    Q = ortho_group.rvs(dim=width)
    A = np.matmul(P, Q.T)
    return A

def generate_incoherent_matrix(height, width, cond_number):
    mat = generate_tall_matrix(height, width, cond_number)+10e-8*np.ones((height, width))
    return mat


def generate_coherent_matrix(height, width, cond_number: float, eigenmin: float = 1):
    low = eigenmin
    high = eigenmin * cond_number
    eigen = np.linspace(low, high, width)
    LAM = np.zeros((width, width))
    np.fill_diagonal(LAM, eigen)
    mat = np.concatenate((LAM, np.zeros((height-width, width))))+10e-8*np.ones((height, width))
    return mat


def generate_semi_coherent_matrix(m, p, cond_number):
    B = generate_incoherent_matrix(m - p // 2, p // 2, np.sqrt(cond_number))
    A1 = np.concatenate((B, np.zeros((m - p // 2, p-p // 2))), 1)
    A2 = np.concatenate((np.zeros((p // 2, p-p // 2)), np.diag(np.ones(p // 2))), 1)
    mat = np.concatenate((A1, A2)) + 10e-8 * np.ones((m, p))
    print('semicoherent', np.linalg.cond(mat))
    return mat


def generate_tensor(height: int, width: int, depth: int, coh: str, cond_number: float, horizontal_mix: bool = False):
    if coh == 'incoherent':
        gen_mat_fun = generate_incoherent_matrix
    elif coh == 'coherent':
        gen_mat_fun = generate_coherent_matrix
    elif coh == 'semi-coherent':
        gen_mat_fun = generate_semi_coherent_matrix
    synth_tensor = np.empty((height, width, 0))
    for i in range(depth):
        slice_i = np.expand_dims(gen_mat_fun(height, width, cond_number), -1)
        if horizontal_mix:
            np.random.shuffle(slice_i)
        synth_tensor = np.concatenate((synth_tensor, slice_i), 2)
    return synth_tensor

def generate_coh_cond_system(coh, cond_number, horizontal_mix, height=1000, width=20, depth=10):
    A_rec_tr = generate_tensor(height, width, depth, coh, cond_number, horizontal_mix)
    omatB_tr = np.random.randn(height, 1, depth)
    funM, invM = generate_haar(depth, 21)
    tenA = invM(A_rec_tr)
    omatB = invM(omatB_tr)
    return tenA, omatB, funM, invM
