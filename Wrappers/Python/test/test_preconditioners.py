import matplotlib.pyplot as plt
from cil.utilities.display import show2D
from cil.recon import FDK
from cil.optimisation.algorithms import  GD, SIRT
from cil.optimisation.functions import  LeastSquares
from cil.framework import ImageGeometry
from cil.optimisation.operators import IdentityOperator

from cil.optimisation.utilities import  Sensitivity
import numpy as np

from testclass import CCPiTestClass

class TestPreconditioners(CCPiTestClass):
    
    def test_preconditioners_init(self):
        #TODO:
        pass
    
    def test_sensitivity(self):
        
        ig = ImageGeometry(12,13,14)
        data = ig.allocate('random')
        A= IdentityOperator(ig)


        sirt = SIRT(ig.allocate(0), A, data,   update_objective_interval=1)
        sirt.run(10)
        
        M = 1./A.direct(A.domain_geometry().allocate(value=1.0)) 
        f = LeastSquares(A=A, b=data, c=0.5, weight=M)
        step_size = 1.
        preconditioner = Sensitivity(A)
        def update_objective(self):
            # SIRT computes |Ax_{k} - b|_2^2
            # GD with weighted LeastSquares computes the objective included the weight, so we remove the weight
            self.loss.append(0.5*(self.objective_function.A.direct(self.x) - self.objective_function.b).squared_norm())
            
        GD.update_objective = update_objective
        precond_pwls = GD(initial=ig.allocate(0), objective_function=f,   preconditioner = preconditioner,
               max_iteration=100, update_objective_interval=1, step_size = step_size)       

        precond_pwls.run(10)
        np.testing.assert_allclose(sirt.solution.array, precond_pwls.solution.array, atol=1e-4)