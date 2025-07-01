#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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

from cil.optimisation.functions import Function, BlockFunction


class ScaledArgFunction(Function):
    r""" ScaledArgFunction represents the scaling of the argument of a function F with a scalar.

    Let a function F and consider :math:`G(x) = F(\alpha x)`.

    If :math:`G(x) = F(\alpha x)` then:

    1. :math:`G(x) = F(\alpha x)` ( __call__ method )
    2. :math:`G'(x) = \alpha F'(\alpha x)` ( gradient method )
    3. :math:`G^{*}(x^{*}) = F^{*}(\frac{x^{*}}{\alpha})` ( convex_conjugate method )
    4. :math:`\text{prox}_{\tau G}(x) = \frac{1}{\alpha}\text{prox}_{\tau\alpha^{2} F}(\alpha x)` ( proximal method )

    Parameters
    ----------
    function: Function or BlockFunction
        The function or BlockFunction to scale the argument of.
        If a BlockFunction is provided, the scalar can be a number or a list of numbers with the same length as the
        number of functions in the BlockFunction.
    scalar: number or list of numbers
        The scalar to scale the argument with. If a list is provided, it must have the same length as the number
        of functions in a BlockFunction.
    Returns
    -------
    ScaledArgFunction or BlockFunction
        A new _ScaledArgFunction or BlockFunction with the argument scaled by the scalar.
    """

    def __init__(self, function, scalar):
        from cil.optimisation.functions import BlockFunction
        super(ScaledArgFunction, self).__init__(L=None)
        if isinstance(function, BlockFunction):
            if isinstance(scalar, (int, float)):
                for i in range(functions.length):
                    scaled_functions = [
                        ScaledArgFunction(function.functions[i], scalar)
                        for i in range(function.length)
                    ]
                self.function = BlockFunction(*scaled_functions)
            elif len(scalar) == function.length:
                scaled_functions = [
                    ScaledArgFunction(function.functions[i], scalar[i])
                    for i in range(function.length)
                ]
                self.function = BlockFunction(*scaled_functions)

            else:
                raise ValueError(
                    "Scalar must be a number or a list of numbers with the same length as the number of functions in BlockFunction.")
        else:
            self.function = _ScaledArgFunction(function, scalar)

    def __call__(self, x):
        """Returns the value of the scaled function evaluated at :math:`x`.
        """
        return self.function(x)

    def gradient(self, x, out=None):
        """Returns the gradient of the scaled function evaluated at :math:`x`.

        Parameters
        ----------
        x : DataContainer  
        out: return DataContainer, if None a new DataContainer is returned, default None.

        Returns
        -------
        DataContainer, the gradient of the scaled function evaluated at :math:`x`.
        """
        return self.function.gradient(x, out=out)

    def proximal(self, x, tau, out=None):
        """Returns the proximal operator of the scaled function evaluated at :math:`x`.

        Parameters
        ----------
        x : DataContainer  
        tau: scalar
        out: return DataContainer, if None a new DataContainer is returned, default None.

        Returns
        -------
        DataContainer, the proximal operator of the scaled function evaluated at :math:`x` with scalar :math:`\tau`.
        """
        return self.function.proximal(x, tau, out=out)

    def convex_conjugate(self, x):
        """Returns the convex conjugate of the scaled function evaluated at :math:`x`.

        Parameters
        ----------
        x : DataContainer

        Returns
        -------
        The value of the convex conjugate of the scaled function at :math:`x`.
        """
        return self.function.convex_conjugate(x)


class _ScaledArgFunction(Function):
    def __init__(self, function, scalar):
        r""" ScaledArgFunction represents the scaling of the argument of a function F with a scalar.

        Let a function F and consider :math:`G(x) = F(\alpha x)`.

        If :math:`G(x) = F(\alpha x)` then:

        1. :math:`G(x) = F(\alpha x)` ( __call__ method )
        2. :math:`G'(x) = \alpha F'(\alpha x)` ( gradient method )
        3. :math:`G^{*}(x^{*}) = F^{*}(\frac{x^{*}}{\alpha})` ( convex_conjugate method )
        4. :math:`\text{prox}_{\tau G}(x) = \frac{1}{\apha}\text{prox}_{\tau\alpha^{2} F}(\alpha x)` ( proximal method )

        Parameters
        ----------
        function: Function
            The function to scale the argument of.
        scalar: number
            The scalar to scale the argument with.
        """

        self.function = function
        self.scalar = scalar
        super(_ScaledArgFunction, self).__init__(
            L=function.L * scalar if function.L is not None else None)

    def __call__(self, x):
        """Returns the value of the scaled function evaluated at :math:`x`.
        """
        x *= self.scalar
        ret = self.function(x)
        x /= self.scalar
        return ret

    def gradient(self, x, out=None):
        x *= self.scalar

        if out is None:
            out = self.function.gradient(x)
        else:
            self.function.gradient(x, out=out)
        out *= self.scalar
        x /= self.scalar

        return out

    def proximal(self, x, tau, out=None):
        """Returns the proximal operator of the scaled function evaluated at :math:`x`.

        .. math:: \text{prox}_{\tau G}(x) = \frac{1}{\alpha}\text{prox}_{\tau\alpha^{2} F}(\alpha x)

        Parameters
        ----------
        x : DataContainer  
        tau: scalar
        out: return DataContainer, if None a new DataContainer is returned, default None.

        Returns
        -------
        DataContainer, the proximal operator of the scaled function evaluated at :math:`x` with scalar :math:`\tau`.

        References
        ----------
        Eq 6.6 of https://archive.siam.org/books/mo25/mo25_ch6.pdf
        """

        if out is None:
            out = x * 0

        x *= self.scalar
        self.function.proximal(x, tau * self.scalar**2, out=out)
        x /= self.scalar
        out /= self.scalar

        return out

    def convex_conjugate(self, x):
        """Returns the convex conjugate of the scaled function evaluated at :math:`x`.

        .. math:: G^{*}(x^{*}) = F^{*}(\frac{x^{*}}{\alpha})

        Parameters
        ----------
        x : DataContainer

        Returns
        -------
        The value of the convex conjugate of the scaled function at :math:`x`.

        References
        ----------
        https://en.wikipedia.org/wiki/Convex_conjugate#Table_of_selected_convex_conjugates

        """
        x /= self.scalar
        ret = self.function.convex_conjugate(x)
        x *= self.scalar

        return ret
