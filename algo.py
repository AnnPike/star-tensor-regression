import numpy as np
from einsumt import einsumt as einsum  #Multithreaded version of numpy.einsum function
from typing import Callable, Optional

NumpynDArray = np.ndarray
MatrixTensorProduct = Callable[[NumpynDArray], NumpynDArray]

TOL = 10 ** -9


class StarAlgebra:
    """ """

    def __init__(self, transforM: MatrixTensorProduct, inv_transforM: MatrixTensorProduct):
        self.transforM = transforM
        self.inv_transforM = inv_transforM

    def fitCG_predict(self, tenA: NumpynDArray, omatB: NumpynDArray, num_iter: Optional[int] = None,
                      normalize: bool = False):
        """

        Args:
          tenA: NumpynDArray:
          omatB: NumpynDArray:
          num_iter: Optional[int]:  (Default value = None)

        Returns:

        """
        self.height, self.width, self.depth = self._dimensionality_assertion_CG(tenA, omatB, square=True)
        tenA_tr = self.transforM(tenA)
        omatB_tr = self.transforM(omatB)
        if num_iter is None:
            num_iter = self.height

        #initialize
        X = np.zeros((self.width, 1, self.depth))
        self.iterative_solutions_CG = [X]

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
            self.iterative_solutions_CG.append(X)
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
        self.height, self.width, self.depth = self._dimensionality_assertion_CG(tenA, omatB, square=False)
        if num_iter is None:
            num_iter = self.width
        tenA_tr = self.transforM(tenA)
        omatB_tr = self.transforM(omatB)
        if tenP:
            tenP_tr = self.transforM(tenP)
        else:
            tenP_tr = self.unit_tensor_transform(self.width, self.depth)
        X = np.zeros((self.width, 1, self.depth))
        self.iterative_solutions_LSQR = [X]

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
            self.iterative_solutions_LSQR.append(X)
            W_wave = V_wave - tao / ro * W_wave
        X = self.inv_transforM(X)
        return X

    def unit_tensor_transform(self, size, depth):
        return np.concatenate([np.eye(size).reshape(size, size, 1) for d in range(depth)], axis=2)

    def normalize(self,
                  omat: NumpynDArray,
                  transforM: Optional[MatrixTensorProduct] = None,
                  inv_transforM: Optional[MatrixTensorProduct] = None
                  ) -> tuple[NumpynDArray, NumpynDArray]:
        """

        Args:
          omat: NumpynDArray: 
          transforM: Optional[MatrixTensorProduct]:  (Default value = None)
          inv_transforM: Optional[MatrixTensorProduct]:  (Default value = None)

        Returns:

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
        """Performs fast (parallel) multiplication of corresopnding frontal slices of tenA and tenB

        Args:
          tenA: NumpynDArray
        tensor of shape m,p,n
          tenB: NumpynDArray
        tensor of shape p,l,n

        Returns:

        
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

        Args:
          ten: NumpynDArray: 

        Returns:

        """
        return np.sqrt((ten ** 2).sum())

    @staticmethod
    def Mnorm(
            omat: NumpynDArray,
            transforM: Optional[MatrixTensorProduct] = None,
            inv_transforM: Optional[MatrixTensorProduct] = None):
        """

        Args:
          omat: NumpynDArray: 
          transforM: Optional[MatrixTensorProduct]:  (Default value = None)
          inv_transforM: Optional[MatrixTensorProduct]:  (Default value = None)

        Returns:

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

        Args:
          alpha: NumpynDArray: 
          transforM: Optional[MatrixTensorProduct]:  (Default value = None)
          inv_transforM: Optional[MatrixTensorProduct]:  (Default value = None)
          tol: float:  (Default value = TOL)

        Returns:

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

    def _dimensionality_assertion_CG(self, tenA: NumpynDArray, omatB: NumpynDArray, square: bool):
        """Fast implementation of Star CG - performed in transformed space, and the result is returned to original space

        Args:
          tenA(np.array :): 
          omatB(np.array :): 
          tenA: NumpynDArray: 
          omatB: NumpynDArray: 

        Returns:

        
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

        # target tubal vector assert
        if omatB.ndim != 3:
            if omatB.ndim == 2:
                height_target, width_target = omatB.shape
                depth_target = 1
            else:
                raise Exception("This method solves only for target omatB is matrix or tensor with width 1"
                                f"Got dimension: {omatB.ndim}, EXPECTED: 2 or 3")
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
