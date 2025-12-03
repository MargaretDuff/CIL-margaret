#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# - CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt


from abc import ABC, abstractmethod
import numpy as np


class Preconditioner(ABC):

    r"""
    Abstract base class for Preconditioner objects. The `apply` method of this class takes an initialised CIL function as an argument and modifies a provided  `gradient`.


    Methods
    -------
    apply(x)
        Abstract method to call the preconditioner.
    """

    def __init__(self):
        '''Initialises the preconditioner
        '''
        pass

    @abstractmethod
    def apply(self, algorithm, gradient, out):
        r"""
        Abstract method to apply the preconditioner.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm object.
        gradient : DataContainer
            The calculated gradient to modify
        out : DataContainer, 
            Container to fill with the modified gradient. 

        Note
        -----

        In CIL algorithms, the preconditioners are used in-place. Make sure this method is safe to use in place. 

        Returns
        -------
        DataContainer
            The modified gradient

        """
        pass


class Sensitivity(Preconditioner):

    r"""
    Sensitivity preconditioner class. 

    In each call to the preconditioner the `gradient` is multiplied by :math:`1/(A^T \mathbf{1})` where :math:`A` is an operator, :math:`\mathbf{1}` is an object in the range of the operator filled with ones.

    Parameters
    ----------
    operator : CIL Operator 
        The operator used for sensitivity computation.

    """

    def __init__(self, operator):

        super(Sensitivity, self).__init__()
        self.operator = operator

        self.compute_preconditioner_matrix()

    def compute_preconditioner_matrix(self):
        r"""
        Compute the sensitivity. :math:`A^T \mathbf{1}` where :math:`A` is the operator and :math:`\mathbf{1}` is an object in the range of the operator filled with ones.        
        Then perform safe division by the sensitivity to store the preconditioner array :math:`1/(A^T \mathbf{1})`
        """
        self.array = self.operator.adjoint(
            self.operator.range_geometry().allocate(value=1.0))

        try:
            self.operator.range_geometry().allocate(value=1.0).divide(
                self.array, where=np.abs(self.array.as_array()) > 0, out=self.array)
        except:  # Due to CIL/SIRF compatibility and SIRF divide not taking kwargs
            sensitivity_np = self.array.as_array()
            self.pos_ind = np.abs(sensitivity_np) > 0
            array_np = 0*sensitivity_np
            array_np[self.pos_ind] = (1./sensitivity_np[self.pos_ind])
            self.array.fill(array_np)

    def apply(self, algorithm, gradient, out=None):
        r"""
        Update the preconditioner.

        Parameters
        ----------
        algorithm : object
            The algorithm object.
        gradient : DataContainer
            The calculated gradient to modify
        out : DataContainer, 
            Container to fill with the modified gradient 

        Returns
        -------
        DataContainer
            The modified gradient

        """
        if out is None:
            out = gradient.copy()
        gradient.multiply(
            self.array, out=out)
        return out


class AdaptiveSensitivity(Sensitivity):

    r"""
    Adaptive Sensitivity preconditioner class. 

    In each call to the preconditioner the `gradient` is multiplied by :math:`(x+\delta) /(A^T \mathbf{1})` where :math:`A` is an operator,  :math:`\mathbf{1}` is an object in the range of the operator filled with ones.
    The point :math:`x` is the current iteration, or a reference image,  and :math:`\delta` is a small positive float. 


    Parameters
    ----------
    operator : CIL object
        The operator used for sensitivity computation.
    delta : float, optional
        The delta value for the preconditioner.
    reference : DataContainer e.g. ImageData, default is None
        Reference data, an object in the domain of the operator. Recommended to be a best guess reconstruction. If reference data is passed the preconditioner is always fixed.   
    max_iterations : int,  default = 100
        The maximum number of iterations before the preconditoner is frozen and no-longer updates. Note that if reference data is passed the preconditioner is always frozen and `iterations` is set to -1. 

    Note
    ----
    A reference for the freezing of the preconditioner can be found: Twyman R., Arridge S., Kereta Z., Jin B., Brusaferri L., Ahn S., Stearns CW., Hutton B.F., Burger I.A., Kotasidis F., Thielemans K.. An Investigation of Stochastic Variance Reduction Algorithms for Relative Difference Penalized 3D PET Image Reconstruction. IEEE Trans Med Imaging. 2023 Jan;42(1):29-41. doi: 10.1109/TMI.2022.3203237. Epub 2022 Dec 29. PMID: 36044488.

    """

    def __init__(self, operator, delta=1e-6, max_iterations=100, reference=None):

        self.max_iterations = max_iterations
        self.delta = delta

        super(AdaptiveSensitivity, self).__init__(
            operator=operator)

        self.freezing_point = operator.domain_geometry().allocate(0)
        if reference is not None:
            reference += self.delta
            self.array.multiply(reference, out=self.freezing_point)
            reference -= self.delta
            self.max_iterations = -1

    def apply(self, algorithm, gradient, out=None):
        r"""
        Update the preconditioner.

        Parameters
        ----------
        algorithm : object
            The algorithm object.
        gradient : DataContainer
            The calculated gradient to modify
        out : DataContainer, 
            Container to fill with the modified gradient 

        Returns
        -------
        DataContainer
            The modified gradient
        """
        if out is None:
            out = gradient.copy()

        if algorithm.iteration <= self.max_iterations:
            self.freezing_point.fill(algorithm.solution)
            self.freezing_point.add(self.delta, out=self.freezing_point)
            self.array.multiply(self.freezing_point,
                                out=self.freezing_point)
            gradient.multiply(
                self.freezing_point, out=out)
        else:
            gradient.multiply(
                self.freezing_point, out=out)

        return out
    
class AdaGrad(Preconditioner):

    """
    This Adaptive Gradient method multiplies the gradient, :math`\nabla f(x_k)`, by :math:`1/\sqrt{ s_k^2 +\epislon}`. Where :math:`s_k^2=s^2_{k-1}+diag((\nabla f(x_k))(\nabla f(x_k))^T)``. 
    
    Reference
    ---------
    Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7).
    """
    
    
    def __init__(self, epsilon=1e-8):
        self.epsilon=epsilon  
        self.gradient_accumulator=None

    def apply(self, algorithm, gradient, out=None):
        """
        Method to __call__ the preconditioner. This multiplies the gradient, :math`\nabla f(x_k)`, by :math:`1/\sqrt{ s_k^2 +\epislon}`. Where :math:`s_k^2=s^2_{k-1}+diag((\nabla f(x_k))(\nabla f(x_k))^T)`. 

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm object.
        gradient : DataContainer
            The calculated gradient to modify   
        out : DataContainer, 
            Container to fill with the modified gradient
        """    
              
        if out is None:
            out = gradient.copy()
            
        if self.gradient_accumulator is None:
            self.gradient_accumulator = gradient.multiply(gradient)
            
        
        else: 
            self.gradient_accumulator.add(  gradient.multiply(gradient), out=self.gradient_accumulator)
            

        gradient.divide(( self.gradient_accumulator+self.epsilon).sqrt(), out=out)

        
class Adam(Preconditioner):

    """
    This ADAM method combines the adaptive learning rate of AdaGrad with the idea of moomentum. 
    
    ADAM keeps track of the momentum :math:`g_k=\gamma g_{k-1} +(1-\gamma)\nabla f(x_k).
    
    It also updates the scaling factors  :math:`s_k^2=\beta s^2_{k-1}+(1-\beta) diag((\nabla f(x_k))(\nabla x_k)^T)`. 
    
    The returned `new` gradient is :math:`g_k/\sqrt(s_k^2+\epsilon)`

    
    Reference
    ---------
    Kingma, D.P. and Ba, J., 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

    """
    
    
    def __init__(self, gamma=0.9, beta =0.999, epsilon=1e-8):
        self.gamma=gamma
        self.beta=beta 
        self.gradient_accumulator=None
        self.scaling_factor_accumulator=None
        self.epsilon=epsilon 

    def apply(self, algorithm, gradient, out=None):
        """
        Method to __call__ the preconditioner, updating `self.x_update` with the preconditioned gradient.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm object.
        gradient : DataContainer
            The calculated gradient to modify
        out : DataContainer, 
            Container to fill with the modified gradient
        """    
        if out is None:
            out = gradient.copy()
        
        if self.gradient_accumulator is None:
            self.gradient_accumulator = gradient.copy()
            self.scaling_factor_accumulator=  gradient.multiply(gradient)
        
        else: 
            self.gradient_accumulator.sapyb(self.gamma, gradient, (1-self.gamma), out=self.gradient_accumulator)
            self.scaling_factor_accumulator.sapyb(self.beta,  gradient.multiply(gradient), (1-self.beta), out=self.scaling_factor_accumulator)
        
        self.gradient_accumulator.divide(( self.scaling_factor_accumulator+self.epsilon).sqrt(), out=out)
        
        
          
