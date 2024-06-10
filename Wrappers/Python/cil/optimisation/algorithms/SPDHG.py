#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
# Claire Delplancke (University of Bath)

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.operators import BlockOperator
import numpy as np
import logging
from cil.optimisation.utilities import Sampler
from numbers import Number
import numpy as np


log = logging.getLogger(__name__)


class SPDHG(Algorithm):
    r'''Stochastic Primal Dual Hybrid Gradient (SPDHG) solves separable optimisation problems of the type: 

    Problem: 


    .. math::

      \min_{x} f(Kx) + g(x) = \min_{x} \sum f_i(K_i x) + g(x)

    Parameters
    ----------
    f : BlockFunction
        Each must be a convex function with a "simple" proximal method of its conjugate
    g : Function
        A convex function with a "simple" proximal
    operator : BlockOperator
        BlockOperator must contain Linear Operators
    tau : positive float, optional, default=None
        Step size parameter for Primal problem
    sigma : list of positive float, optional, default=None
        List of Step size parameters for Dual problem
    initial : DataContainer, optional, default=None
        Initial point for the SPDHG algorithm
    gamma : float
        parameter controlling the trade-off between the primal and dual step sizes
    sampler: an instance of a `cil.optimisation.utilities.Sampler` class or another class with the function __next__(self) implemented outputting an integer from {1,...,len(operator)}. 
             Method of selecting the next index for the SPDHG update. If None, a sampler will be created for random sampling with replacement and each index will have probability = 1/len(operator)


    **kwargs:
    prob : list of floats, optional, default=None
        List of probabilities. If None each subset will have probability = 1/number of subsets. To be deprecated.
    norms : list of floats
        Precalculated list of norms of the operators. To be deprecated and placed by the `set_norms` functionalist in a BlockOperator.


    Example
    -------
    >>> data = dataexample.SIMPLE_PHANTOM_2D.get(size=(20, 20))
    >>> subsets = 10
    >>> ig = data.geometry
    >>> ig.voxel_size_x = 0.1
    >>> ig.voxel_size_y = 0.1
    >>> 
    >>> detectors = ig.shape[0]
    >>> angles = np.linspace(0, np.pi, 90)
    >>> ag = AcquisitionGeometry.create_Parallel2D().set_angles(angles, angle_unit='radian').set_panel(detectors, 0.1)
    >>> 
    >>> Aop = ProjectionOperator(ig, ag, 'cpu')
    >>> 
    >>> sin = Aop.direct(data)
    >>> partitioned_data = sin.partition(subsets, 'sequential')
    >>> A = BlockOperator(*[ProjectionOperator(ig. partitioned_data[i].geometry, 'cpu') for i in range(subsets)])
    >>> 
    >>> F = BlockFunction(*[L2NormSquared(b=partitioned_data[i])
                            for i in range(subsets)])

    >>> alpha = 0.025
    >>> G = alpha * TotalVariation()
    >>> spdhg = SPDHG(f=F, g=G, operator=A, sampler=Sampler.sequential(len(A)),
                      initial=A.domain_geometry().allocate(1), max_iteration=1000, update_objective_interval=10)
    >>> spdhg.run(100)


    Example
    -------
    Further examples of usage see the [CIL demos.](https://github.com/vais-ral/CIL-Demos/blob/master/Tomography/Simulated/Single%20Channel/PDHG_vs_SPDHG.py)

    Note
    -----
    When setting `sigma` and `tau`, there are 4 possible cases considered by setup function: 

    - Case 1: If neither `sigma` or `tau` are provided then `sigma` is set using the formula:

        .. math:: 
          \sigma_i=0.99 / (\|K_i\|**2)

        and `tau` is set as per case 2

    - Case 2: If `sigma` is provided but not `tau` then `tau` is calculated using the formula 

        .. math:: 
        \tau = 0.99\min_i([p_i / (\sigma_i * \|K_i\|**2) ])

    - Case 3: If `tau` is provided but not `sigma` then `sigma` is calculated using the formula

        .. math:: 
        \sigma_i=0.99 p_i / (\tau*\|K_i\|**2)

    - Case 4: Both `sigma` and `tau` are provided.


    Note
    ----

    Convergence is guaranteed provided that [2, eq. (12)]:

    .. math:: 

    \|\sigma[i]^{1/2} * K[i] * tau^{1/2} \|^2  < p_i for all i

    References
    ----------

    [1]"Stochastic primal-dual hybrid gradient algorithm with arbitrary 
    sampling and imaging applications",
    Chambolle, Antonin, Matthias J. Ehrhardt, Peter Richtárik, and Carola-Bibiane Schonlieb,
    SIAM Journal on Optimization 28, no. 4 (2018): 2783-2808.   

    [2]"Faster PET reconstruction with non-smooth priors by randomization and preconditioning",
    Matthias J Ehrhardt, Pawel Markiewicz and Carola-Bibiane Schönlieb,
    Physics in Medicine & Biology, Volume 64, Number 22, 2019.
    '''

    def __init__(self, f=None, g=None, operator=None, tau=None, sigma=None,
                 initial=None, sampler=None, prob_weights=None,   **kwargs):

        max_iteration = kwargs.pop('max_iteration', 0)

        print_interval = kwargs.pop('print_interval', None)
        log_file = kwargs.pop('log_file', None)
        update_objective_interval = kwargs.pop('update_objective_interval', 1)
        super(SPDHG, self).__init__(max_iteration=max_iteration,
                                    update_objective_interval=update_objective_interval, log_file=log_file, print_interval=print_interval)

        self.set_up(f=f, g=g, operator=operator, sigma=sigma, tau=tau,
                    initial=initial,  sampler=sampler, prob_weights=prob_weights,  **kwargs)

    def set_up(self, f, g, operator, sigma=None, tau=None,
               initial=None,   sampler=None, prob_weights=None, **deprecated_kwargs):

        '''set-up of the algorithm

        Parameters
        ----------
        f : BlockFunction
            Each must be a convex function with a "simple" proximal method of its conjugate
        g : Function
            A convex function with a "simple" proximal
        operator : BlockOperator
            BlockOperator must contain Linear Operators
        tau : positive float, optional, default=None
            Step size parameter for Primal problem
        sigma : list of positive float, optional, default=None
            List of Step size parameters for Dual problem
        initial : DataContainer, optional, default=None
            Initial point for the SPDHG algorithm
        gamma : float
            parameter controlling the trade-off between the primal and dual step sizes
        sampler: an instance of a `cil.optimisation.utilities.Sampler` class or another class with the function __next__(self) implemented outputting a sample from {1,...,len(operator)}. 
             Method of selecting the next index for the SPDHG update. If None, a sampler will be created for random sampling  with replacement and each index will have probability = 1/len(operator)
        prob_weights: list of floats of length num_indices that sum to 1. Defaults to [1/len(operator)]*len(operator)
            Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. Note that this should not be passed if the provided sampler has it as an attribute. 

        '''
        log.info("%s setting up", self.__class__.__name__)
    
        # algorithmic parameters
        self.f = f
        self.g = g
        self.operator = operator

        if not isinstance(operator, BlockOperator):
            raise TypeError("operator should be a BlockOperator")

        self._ndual_subsets = len(self.operator)
        self._sampler = sampler

        self._prob_weights = getattr(self._sampler, 'prob_weights', None)
        if prob_weights is not None:
            if self._prob_weights is None:
                self._prob_weights = prob_weights
            else:
                raise ValueError(
                    ' You passed a `prob_weights` argument and a sampler with attribute `prob_weights`, please remove the `prob_weights` argument.')

        self._deprecated_kwargs(deprecated_kwargs)

        if self._prob_weights is None:
            self._prob_weights = [1/self._ndual_subsets]*self._ndual_subsets

        if self._sampler is None:
            self._sampler = Sampler.random_with_replacement(
                len(operator), prob=self._prob_weights)

        self._norms = operator.get_norms_as_list()

        self.set_step_sizes(sigma=sigma, tau=tau)

        # initialize primal variable
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()


        self._x_tmp = self.operator.domain_geometry().allocate(0)

        # initialize dual variable to 0
        self._y_old = operator.range_geometry().allocate(0)

        # initialize variable z corresponding to back-projected dual variable
        self._z = operator.domain_geometry().allocate(0)
        self._zbar = operator.domain_geometry().allocate(0)
        # relaxation parameter
        self._theta = 1
        
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))

    def _deprecated_kwargs(self, deprecated_kwargs):
        """
        Handle deprecated keyword arguments for backward compatibility.

        Parameters
        ----------
        deprecated_kwargs : dict
            Dictionary of keyword arguments.

        Notes
        -----
        This method is called by the set_up method.
        """
        norms = deprecated_kwargs.pop('norms', None)
        prob = deprecated_kwargs.pop('prob', None)

        if prob is not None:
            if self._prob_weights is None:
                warnings.warn('`prob` is being deprecated to be replaced with a sampler class and `prob_weights`. To randomly sample with replacement use "sampler=Sampler.randomWithReplacement(number_of_subsets,  prob=prob). To pass probabilites to the calculation for `sigma` and `tau` please use `prob_weights`. ')
                self._prob_weights = prob
            else:

                raise ValueError(
                    '`prob` is being deprecated to be replaced with a sampler class and `prob_weights`. You passed  a `prob` argument, and either a `prob_weights` argument or a sampler with a `prob_weights` property. Please give only one of the three. ')

        if norms is not None:
            self.operator.set_norms(norms)
            warnings.warn(
                ' `norms` is being deprecated, use instead the `BlockOperator` function `set_norms`')

        if deprecated_kwargs:
            raise ValueError("Additional keyword arguments passed but not used: {}".format(
                deprecated_kwargs))

    @property
    def sigma(self):
        return self._sigma

    @property
    def tau(self):
        return self._tau

    def set_step_sizes_from_ratio(self, gamma=1.0, rho=0.99):
        r""" Sets gamma, the step-size ratio for the SPDHG algorithm. Currently gamma takes a scalar value.

        Parameters
        ----------
            gamma : Positive float
                parameter controlling the trade-off between the primal and dual step sizes
            rho : Positive float
                 parameter controlling the size of the product :math: \sigma\tau :math:

        Note
        -----
        The step sizes `sigma` and `tau` are set using the equations:
        .. math:: 
            \sigma_i=\gamma\rho / (\|K_i\|**2)\\
            \tau = (\rho/\gamma)\min_i([p_i / (\sigma_i * \|K_i\|**2) ])

        """
        if isinstance(gamma, Number):
            if gamma <= 0:
                raise ValueError(
                    "The step-sizes of SPDHG are positive, gamma should also be positive")

        else:
            raise ValueError(
                "We currently only support scalar values of gamma")
        if isinstance(rho, Number):
            if rho <= 0:
                raise ValueError(
                    "The step-sizes of SPDHG are positive, rho should also be positive")

        else:
            raise ValueError(
                "We currently only support scalar values of gamma")

        self._sigma = [gamma * rho / ni for ni in self._norms]
        values = [pi / (si * ni**2) for pi, ni,
                  si in zip(self._prob_weights, self._norms, self._sigma)]
        self._tau = min([value for value in values if value > 1e-8])
        self._tau *= (rho / gamma)

    def set_step_sizes(self, sigma=None, tau=None):
        r""" Sets sigma and tau step-sizes for the SPDHG algorithm after the initial set-up. The step sizes can be either scalar or array-objects.

        Parameters
        ----------
            sigma : list of positive float, optional, default=None
                List of Step size parameters for Dual problem
            tau : positive float, optional, default=None
                Step size parameter for Primal problem

        The user can set these or default values are calculated, either sigma, tau, both or None can be passed. 

        Note
        -----
        When setting `sigma` and `tau`, there are 4 possible cases considered by setup function: 

        - Case 1: If neither `sigma` or `tau` are provided then `sigma` is set using the formula:

            .. math:: 
            \sigma_i=0.99 / (\|K_i\|**2)


            and `tau` is set as per case 2

        - Case 2: If `sigma` is provided but not `tau` then `tau` is calculated using the formula 

            .. math:: 
            \tau = 0.99\min_i([p_i / (\sigma_i * \|K_i\|**2) ])

        - Case 3: If `tau` is provided but not `sigma` then `sigma` is calculated using the formula

            .. math:: 
            \sigma_i=0.99 p_i / (\tau*\|K_i\|**2)

        - Case 4: Both `sigma` and `tau` are provided.

        """
        gamma = 1.
        rho = .99
        if sigma is not None:
            if len(sigma) == self._ndual_subsets:
                if all(isinstance(x, Number) and x > 0 for x in sigma):
                    pass
                else:
                    raise ValueError(
                        "Sigma expected to be a positive number.")

            else:
                raise ValueError(
                    "Please pass a list of floats to sigma with the same number of entries as number of operators")
            self._sigma = sigma

        elif tau is None:
            self._sigma = [gamma * rho / ni for ni in self._norms]
        else:
            self._sigma = [
                gamma * rho*pi / (tau*ni**2) for ni, pi in zip(self._norms, self._prob_weights)]

        if tau is None:
            values = [pi / (si * ni**2) for pi, ni,
                      si in zip(self._prob_weights, self._norms, self._sigma)]
            self._tau = min([value for value in values if value > 1e-8])
            self._tau *= (rho / gamma)
        else:
            if isinstance(tau, Number) and tau > 0:
                pass
            else:
                raise ValueError(
                    "The step-sizes of SPDHG must be positive, passed tau = {}".format(tau))

            self._tau = tau

    def check_convergence(self):
        """  Checks whether convergence criterion for SPDHG is satisfied with the current scalar values of tau and sigma

        Returns
        -------
        Boolean
            True if convergence criterion is satisfied. False if not satisfied or convergence is unknown. 
            
        Note
        -----
        Convergence criterion currently can only be checked for scalar values of tau.
        
        Note
        ----
        This checks the convergence criterion. Numerical errors may mean some sigma and tau values that satisfy the convergence criterion may not converge. 
        Alternatively, step sizes outside the convergence criterion may still allow (fast) convergence. 
        """
        for i in range(self._ndual_subsets):
            if isinstance(self._tau, Number) and isinstance(self._sigma[i], Number):
                if self._sigma[i] * self._tau * self._norms[i]**2 > self._prob_weights[i]:
                    return False
                return True
            else:
                return False

    def update(self):
        """  Runs one iteration of SPDHG 

        """
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        self.x.sapyb(1., self._zbar,  -self._tau, out=self._x_tmp)

        self.g.proximal(self._x_tmp, self._tau, out=self.x)

        # Choose subset
        i = next(self._sampler)

        # Gradient ascent for the dual variable
        # y_k = y_old[i] + sigma[i] * K[i] x
        try:
            y_k = self.operator[i].direct(self.x)
        except IndexError:
            raise IndexError(
                'The sampler has outputted an index larger than the number of operators to sample from. Please ensure your sampler samples from {1,2,...,len(operator)} only.')

        y_k.sapyb(self._sigma[i], self._y_old[i], 1., out=y_k)

        y_k = self.f[i].proximal_conjugate(y_k, self._sigma[i])

        # Back-project
        # x_tmp = K[i]^*(y_k - y_old[i])
        y_k.subtract(self._y_old[i], out=self._y_old[i])

        self.operator[i].adjoint(self._y_old[i], out=self._x_tmp)
        # Update backprojected dual variable and extrapolate
        # zbar = z + (1 + theta/p[i]) x_tmp

        # z = z + x_tmp
        self._z.add(self._x_tmp, out=self._z)
        # zbar = z + (theta/p[i]) * x_tmp

        self._z.sapyb(1., self._x_tmp, self._theta /
                     self._prob_weights[i], out=self._zbar)

        # save previous iteration
        self._save_previous_iteration(i, y_k)

    def update_objective(self):
        # p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
        p1 = 0.
        for i, op in enumerate(self.operator.operators):
            p1 += self.f[i](op.direct(self.x))
        p1 += self.g(self.x)

        d1 = - self.f.convex_conjugate(self._y_old)
        tmp = self.operator.adjoint(self._y_old)
        tmp *= -1
        d1 -= self.g.convex_conjugate(tmp)

        self.loss.append([p1, d1, p1-d1])

    @property
    def objective(self):
        '''The saved primal objectives. 
        Returns
        -------
        list
            The saved primal objectives from `update_objective`. The number of saved values depends on the `update_objective_interval` kwarg. 
        '''
        return [x[0] for x in self.loss]

    @property
    def dual_objective(self):
        '''The saved dual objectives. 
        Returns
        -------
        list
            The saved dual objectives from `update_objective`. The number of saved values depends on the `update_objective_interval` kwarg. 
        '''
        return [x[1] for x in self.loss]

    @property
    def primal_dual_gap(self):
        '''The saved primal-dual gap. 
        Returns
        -------
        list
            The saved primal dual gap from `update_objective`. The number of saved values depends on the `update_objective_interval` kwarg. 
        '''
        return [x[2] for x in self.loss]

    def _save_previous_iteration(self, index, y_current):
        ''' Internal function used to save the previous iteration 
        '''
        self._y_old[index].fill(y_current)
