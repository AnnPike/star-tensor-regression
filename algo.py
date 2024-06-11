import numpy as np
from einsumt import einsumt as einsum  #Multithreaded version of numpy.einsum function
from typing import Callable, Optional
from scipy.fft import dct

NumpynDArray = np.ndarray
MatrixTensorProduct = Callable[[NumpynDArray], NumpynDArray]

TOL = 10 ** -9


class StarAlgebra:
    """ """

    def __init__(self, transforM: MatrixTensorProduct, inv_transforM: MatrixTensorProduct):
        self.transforM = transforM
        self.inv_transforM = inv_transforM
        self.iterative_solutions = []

    def sketch(self, tensor: NumpynDArray, s: Optional[int] = None):
        height, width, depth = self._dimensionality_assertion(tensor, omatB=None, square=False)
        ten_hat = self.transforM(tensor)
        d = np.random.choice([-1, 1], height).reshape(height, 1, 1)
        tenDA_hat = d*ten_hat
        tenHDA_hat = dct(tenDA_hat, type=2, n=height, axis=0, norm='ortho', workers=-1)
        if s is None:
            s = 4*width
        chosen_rows = np.random.choice(height, s, replace=False)
        sampled_ten_hat = tenHDA_hat[chosen_rows]
        tensor_sketched = self.inv_transforM(sampled_ten_hat)
        return tensor_sketched

    def fitCG_predict(
            self,
            tenA: NumpynDArray,
            omatB: NumpynDArray,
            num_iter: Optional[int] = None,
            normalize: bool = False):
        """

        Parameters
        ----------
        tenA: NumpynDArray :
            
        omatB: NumpynDArray :
            
        num_iter: Optional[int] :
             (Default value = None)
        normalize: bool :
             (Default value = False)

        Returns
        -------

        """

        height, width, depth = self._dimensionality_assertion(tenA, omatB, square=True)
        tenA_tr = self.transforM(tenA)
        omatB_tr = self.transforM(omatB)
        if num_iter is None:
            num_iter = height

        #initialize
        X = np.zeros((width, 1, depth))
        self.iterative_solutions = [X]

        if normalize:
            R, B_norm_tube = self.normalize(omatB_tr)
        else:
            R = omatB_tr
        D = R.copy()

        for i in range(num_iter):
            alpha_num = self.facewise_mult(R.transpose((1, 0, 2)), R)
            alpha_den = self.facewise_mult(self.facewise_mult(D.transpose((1, 0, 2)), tenA_tr), D)
            alpha = alpha_num * self.tubal_pseudoinverse(alpha_den)

            X = X + alpha * D
            self.iterative_solutions.append(X)
            R_next = R - alpha * self.facewise_mult(tenA_tr, D)

            beta_num = self.facewise_mult(R_next.transpose((1, 0, 2)), R_next)
            beta_den = self.facewise_mult(R.transpose((1, 0, 2)), R)
            beta = beta_num * self.tubal_pseudoinverse(beta_den)
            D = R_next + beta * D
            R = R_next.copy()
        if normalize:
            X = B_norm_tube * X
        X = self.inv_transforM(X)
        return X

    def fit_LSQR_predict(self, tenA, omatB, tenP=None, num_iter: Optional[int] = None):
        """

        Parameters
        ----------
        tenA :
            
        omatB :
            
        tenP :
             (Default value = None)
        num_iter: Optional[int] :
             (Default value = None)

        Returns
        -------

        """

        height, width, depth = self._dimensionality_assertion(tenA, omatB, square=False)
        if num_iter is None:
            num_iter = width
        tenA_tr = self.transforM(tenA)
        omatB_tr = self.transforM(omatB)
        if tenP:
            tenP_tr = self.transforM(tenP)
        else:
            tenP_tr = self.unit_tensor_transform(width, depth)
        X = np.zeros((width, 1, depth))
        self.iterative_solutions = [X]

        U, beta = self.normalize(omatB_tr)

        V, alpha = self.normalize(self.facewise_mult(self.facewise_mult(tenA_tr, tenP_tr).transpose((1, 0, 2)), U))
        V_wave = self.facewise_mult(tenP_tr, V)
        W_wave = V_wave.copy()
        ro_ = alpha.copy()
        fi_ = beta.copy()
        for i in range(num_iter):
            U, beta = self.normalize(self.facewise_mult(tenA_tr, V_wave) - alpha * U)
            V_ = self.facewise_mult(self.facewise_mult(tenA_tr, tenP_tr).transpose((1, 0, 2)), U)
            V, alpha = self.normalize(V_ - beta * V)
            V_wave = self.facewise_mult(tenP_tr, V)

            ro = np.sqrt(ro_ ** 2 + beta ** 2)
            c, s = ro_ / ro, beta / ro
            tao, ro_ = s * alpha, c * alpha
            fi = c * fi_
            fi_ = - s * fi_

            X = X + fi / ro * W_wave
            self.iterative_solutions.append(X)
            W_wave = V_wave - tao / ro * W_wave
        X = self.inv_transforM(X)
        return X

    def solve_normal_Cholesky(self, tenA, omatB, reg: float = 0):
        """

        Parameters
        ----------
        tenA :
            
        omatB :
            
        reg: float :
             (Default value = 0)

        Returns
        -------

        """

        _ = self._dimensionality_assertion(tenA, omatB, square=False)
        tenA_hat = self.transforM(tenA)
        omatB_hat = self.transforM(omatB)
        tenL_hat = self.get_Cholesky(tenA_hat, reg=reg)
        tenL_inv_hat = self.get_inverse_tensor(tenL_hat)
        left_mult = self.facewise_mult(tenA_hat.transpose((1, 0, 2)), omatB_hat)
        mult = self.facewise_mult(tenL_inv_hat, left_mult)
        X_hat = self.facewise_mult(tenL_inv_hat.transpose((1, 0, 2)), mult)
        return self.inv_transforM(X_hat)

    def get_Cholesky(self, tensor,
                  transforM: Optional[MatrixTensorProduct] = None,
                  inv_transforM: Optional[MatrixTensorProduct] = None,
                  reg: float = 0):
        """

        Parameters
        ----------
        tensor :
            
        transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        inv_transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        reg: float :
             (Default value = 0)

        Returns
        -------

        """

        height, width, depth = self._dimensionality_assertion(tensor, omatB=None, square=False)
        if transforM:
            ten_hat = transforM(tensor)
        else:
            ten_hat = tensor.copy()
        gram_ten = self.facewise_mult(ten_hat.transpose(1, 0, 2), ten_hat)
        gram_ten_reg = gram_ten + reg * self.unit_tensor_transform(width, depth)
        tenL_hat = np.linalg.cholesky(gram_ten_reg.transpose(2, 0, 1)).transpose(1, 2, 0)
        if inv_transforM:
            tensorL = inv_transforM(tenL_hat)
        else:
            tensorL = tenL_hat.copy()
        return tensorL

    def solve_normal_QR(self, tenA, omatB):
        _ = self._dimensionality_assertion(tenA, omatB, square=False)
        tenA_hat = self.transforM(tenA)
        omatB_hat = self.transforM(omatB)
        tenQ_hat, tenR_hat = self.get_QR(tenA_hat)
        mult_right = self.facewise_mult(tenQ_hat.transpose((1, 0, 2)), omatB_hat)
        tenR_inv_hat = self.get_inverse_tensor(tenR_hat)
        X_hat = self.facewise_mult(tenR_inv_hat, mult_right)
        X = self.inv_transforM(X_hat)
        return X

    def get_QR(self, tensor,
                     transforM: Optional[MatrixTensorProduct] = None,
                     inv_transforM: Optional[MatrixTensorProduct] = None):
        """

        Parameters
        ----------
        tensor :

        transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        inv_transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        reg: float :
             (Default value = 0)

        Returns
        -------

        """

        height, width, depth = self._dimensionality_assertion(tensor, omatB=None, square=False)
        if transforM:
            ten_hat = transforM(tensor)
        else:
            ten_hat = tensor.copy()

        tenQ_hat_T, tenR_hat_T = np.linalg.qr(ten_hat.transpose(2, 0, 1), mode='reduced')
        tenQ_hat, tenR_hat = tenQ_hat_T.transpose(1, 2, 0), tenR_hat_T.transpose(1, 2, 0)
        if inv_transforM:
            tenQ, tenR = inv_transforM(tenQ_hat), inv_transforM(tenR_hat)
        else:
            tenQ, tenR = tenQ_hat.copy(), tenR_hat.copy()
        return tenQ, tenR

    def get_inverse_tensor(self, tensor, transforM: Optional[MatrixTensorProduct] = None,
                  inv_transforM: Optional[MatrixTensorProduct] = None):
        """

        Parameters
        ----------
        tensor :
            
        transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        inv_transforM: Optional[MatrixTensorProduct] :
             (Default value = None)

        Returns
        -------

        """

        if transforM:
            ten_hat = transforM(tensor)
        else:
            ten_hat = tensor.copy()
        ten_inv_hat = np.linalg.inv(ten_hat.transpose(2, 0, 1)).transpose(1, 2, 0)
        if inv_transforM:
            ten_inv = inv_transforM(ten_inv_hat)
        else:
            ten_inv = ten_inv_hat.copy()
        return ten_inv

    @staticmethod
    def unit_tensor_transform(size, depth):
        """

        Parameters
        ----------
        size :
            
        depth :
            

        Returns
        -------

        """

        return np.concatenate([np.eye(size).reshape(size, size, 1) for d in range(depth)], axis=2)

    def normalize(self,
                  omat: NumpynDArray,
                  transforM: Optional[MatrixTensorProduct] = None,
                  inv_transforM: Optional[MatrixTensorProduct] = None
                  ) -> tuple[NumpynDArray, NumpynDArray]:
        """

        Parameters
        ----------
        omat: NumpynDArray :
            
        transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        inv_transforM: Optional[MatrixTensorProduct] :
             (Default value = None)

        Returns
        -------

        """

        omat_tubal_norm = self.Mnorm(omat, transforM, inv_transforM)
        normalization_tuple = self.tubal_pseudoinverse(omat_tubal_norm, transforM, inv_transforM)
        if transforM is not None:
            omat_tr = transforM(omat)
            normalization_tuple_tr = transforM(normalization_tuple)
        else:
            omat_tr = omat
            normalization_tuple_tr = normalization_tuple
        omat_normalized_tr = normalization_tuple_tr * omat_tr
        if inv_transforM is not None:
            omat_normalized = inv_transforM(omat_normalized_tr)
        else:
            omat_normalized = omat_normalized_tr
        return omat_normalized, omat_tubal_norm

    @staticmethod
    def facewise_mult(tenA, tenB):
        """

        Parameters
        ----------
        tenA :
            
        tenB :
            

        Returns
        -------

        """

        heightA, widthA, depthA = tenA.shape
        heightB, widthB, depthB = tenB.shape
        if widthA != heightB:
            raise Exception("Left tensor width must coincide with right tensor height"
                            f"Got left tensor width: {widthA} NOT EQUAL TO right tensor height: {heightB}")
        if depthA != depthB:
            raise Exception("Left tensor depth must coincide with right tensor depth"
                            f"Got left tensor depth: {depthA} NOT EQUAL TO right tensor depth: {depthB}")

        tenC = einsum('mpi,pli->mli', tenA, tenB)
        return tenC

    @staticmethod
    def Fnorm(ten: NumpynDArray) -> float:
        """

        Parameters
        ----------
        ten: NumpynDArray :
            

        Returns
        -------

        """

        return np.sqrt((ten ** 2).sum())

    @staticmethod
    def Mnorm(
            omat: NumpynDArray,
            transforM: Optional[MatrixTensorProduct] = None,
            inv_transforM: Optional[MatrixTensorProduct] = None):
        """

        Parameters
        ----------
        omat: NumpynDArray :
            
        transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        inv_transforM: Optional[MatrixTensorProduct] :
             (Default value = None)

        Returns
        -------

        """

        if transforM is not None:
            omat_transformed = transforM(omat)
        else:
            omat_transformed = omat.copy()
        tubal_norm_transformed = np.sqrt(einsum('pmi,pli->mli', omat_transformed, omat_transformed))
        if inv_transforM is not None:
            tubal_norm = inv_transforM(tubal_norm_transformed)
        else:
            tubal_norm = tubal_norm_transformed
        return tubal_norm

    @staticmethod
    def tubal_pseudoinverse(
            alpha: NumpynDArray,
            transforM: Optional[MatrixTensorProduct] = None,
            inv_transforM: Optional[MatrixTensorProduct] = None,
            tol: float = TOL):
        """

        Parameters
        ----------
        alpha: NumpynDArray :
            
        transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        inv_transforM: Optional[MatrixTensorProduct] :
             (Default value = None)
        tol: float :
             (Default value = TOL)

        Returns
        -------

        """

        height, width, depth = alpha.shape
        if height != 1 or width != 1 or depth < 2:
            raise Exception("Tubal pseudoinverse is intended only for tubal scalars"
                            f"Got height: {height}, width: {width}, EXPECTED 1"
                            f"Got depth: {depth}, EXPECTED > 1")
        if transforM is not None:
            alpha_transform = transforM(alpha)
        else:
            alpha_transform = alpha
        # pseudo inverse
        alpha_transform_normalized = np.where(alpha_transform < tol, 0, 1 / alpha_transform)
        if inv_transforM is not None:
            alpha_normalized = inv_transforM(alpha_transform_normalized)
        else:
            alpha_normalized = alpha_transform_normalized
        return alpha_normalized

    def _dimensionality_assertion(self, tenA: NumpynDArray, omatB: Optional[NumpynDArray], square: bool):
        """

        Parameters
        ----------
        tenA: NumpynDArray :
            
        omatB: Optional[NumpynDArray] :
            
        square: bool :
            

        Returns
        -------

        """

        # tensor assert
        if tenA.ndim != 3:
            raise Exception("This method solves only for 3 mode tensors. "
                            f"Got dimention: {tenA.ndim}, EXPECTED: 3")
        height, width, depth = tenA.shape
        if square:
            if height != width:
                raise Exception("This method solves only for square tensors. "
                                f"Got height: {height} NOT EQUAL TO width: {width}")
        if omatB is not None:
            # target tubal vector assert
            if omatB.ndim != 3:
                raise Exception("This method solves only for target omatB is matrix or tensor with width 1"
                                    f"Got dimension: {omatB.ndim}, EXPECTED: 3")
            else:
                height_target, width_target, depth_target = omatB.shape
            if width_target != 1:
                raise Exception("This method solves only for target omatB is tubal vector"
                                f"Got width: {width_target}, EXPECTED: 1")

            # compatibility assertions
            if height != height_target:
                raise Exception("Left and right height must be equal. "
                                f"Got tensor height: {height} NOT EQUAL TO target height: {height_target}")
            if depth != depth_target:
                raise Exception("Left and right depth must be equal. "
                                f"Got tensor depth: {depth} NOT EQUAL TO target depth: {depth_target}")
        return height, width, depth
