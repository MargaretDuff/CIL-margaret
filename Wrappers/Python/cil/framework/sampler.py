# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import numpy as np
import math 
import time 

class Sampler():
    
    r"""
    A class to select from a list of integers {0, 1, …, S-1}, with each integer representing the index of a subset
    The function next() outputs a single next index from the {0,1,…,S-1} subset list. Different orders possible incl with and without replacement. To be run again and again, depending on how many iterations.
    
    
    Parameters
    ----------
    num_subsets: int
        The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. 
    
    sampling_type:str
        The sampling type used. 

    order: list of integers
        The list of integers the method selects from using next. 
    
    shuffle= bool, default=False
        If True, after each num_subsets calls of next the sampling order is shuffled randomly. 

    prob: list of floats of length num_subsets that sum to 1. 
        For random sampling with replacement, this is the probability for each integer to be called by next. 

    seed:int, default=None
        Random seed for the methods that use a random number generator.  If set to None, the seed will be set using the current time.  
    


    Example
    -------

    >>> sampler=Sampler.sequential(10)
    >>> print(sampler.get_samples(5))
    >>> for _ in range(11):
            print(sampler.next())

    [0 1 2 3 4]
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    0

    Example
    -------
    >>> sampler=Sampler.randomWithReplacement(5)
    >>> for _ in range(12):
    >>>     print(next(sampler))
    >>> print(sampler.get_samples())
    
    3
    4
    0
    0
    2
    3
    3
    2
    2
    1
    1
    4
    [3 4 0 0 2 3 3 2 2 1 1 4 4 3 0 2 4 4 2 4]



    """
    
    @staticmethod
    def sequential(num_subsets):
        """
        Function that outputs a sampler that outputs sequentially. 

        num_subsets: int
            The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. 

        Example
        -------

        >>> sampler=Sampler.sequential(10)
        >>> print(sampler.get_samples(5))
        >>> for _ in range(11):
                print(sampler.next())

        [0 1 2 3 4]
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        0
        """
        order=list(range(num_subsets))
        sampler=Sampler(num_subsets, sampling_type='sequential', order=order)
        return sampler 
    
    @staticmethod
    def customOrder( customlist):
        """
        Function that outputs a sampler that outputs from a list, one entry at a time before cycling back to the beginning. 

        customlist: list of integers
            The list that will be sampled from in order. 

        Example
        --------

        >>> sampler=Sampler.customOrder([1,4,6,7,8,9,11])
        >>> print(sampler.get_samples(11))
        >>> for _ in range(9):
        >>>     print(sampler.next())
        >>> print(sampler.get_samples(5)) 

        [ 1  4  6  7  8  9 11  1  4  6  7]
        1
        4
        6
        7
        8
        9
        11
        1
        4
        [1 4 6 7 8]

        """
        num_subsets=len(customlist)
        sampler=Sampler(num_subsets, sampling_type='custom_order', order=customlist)
        return  sampler

    @staticmethod
    def hermanMeyer(num_subsets):
        """
        Function that takes a number of subsets and returns a sampler which outputs a Herman Meyer order 

        num_subsets: int
            The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. For Herman-Meyer sampling this number should not be prime. 

        Reference
        ----------
        Herman GT, Meyer LB. Algebraic reconstruction techniques can be made computationally efficient. IEEE Trans Med Imaging.  doi: 10.1109/42.241889.
        
        Example
        -------
        >>> sampler=Sampler.hermanMeyer(12)
        >>> print(sampler.get_samples(16))
        
        [ 0  6  3  9  1  7  4 10  2  8  5 11  0  6  3  9]

        """
        def _herman_meyer_order(n):
            # Assuming that the subsets are in geometrical order
            n_variable = n
            i = 2
            factors = []
            while i * i <= n_variable:
                if n_variable % i:
                    i += 1
                else:
                    n_variable //= i
                    factors.append(i)
            if n_variable > 1:
                factors.append(n_variable)
            n_factors = len(factors)
            if n_factors==0:
                raise ValueError('Herman Meyer sampling defaults to sequential ordering if the number of subsets is prime. Please use an alternative sampling method or change the number of subsets. ')
            order =  [0 for _ in range(n)]
            value = 0
            for factor_n in range(n_factors):
                n_rep_value = 0
                if factor_n == 0:
                    n_change_value = 1
                else:
                    n_change_value = math.prod(factors[:factor_n])
                for element in range(n):
                    mapping = value
                    n_rep_value += 1
                    if n_rep_value >= n_change_value:
                        value = value + 1
                        n_rep_value = 0
                    if value == factors[factor_n]:
                        value = 0
                    order[element] = order[element] + math.prod(factors[factor_n+1:]) * mapping
            return order

        order=_herman_meyer_order(num_subsets)
        sampler=Sampler(num_subsets, sampling_type='herman_meyer', order=order)
        return sampler 

    @staticmethod
    def staggered(num_subsets, offset):

        """
        Function that takes a number of subsets and returns a sampler which outputs in a staggered order. 

        num_subsets: int
            The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. 

        offset: int
            The sampler will output in the order {0, a, 2a, 3a, ...., 1, 1+a, 1+2a, 1+3a,...., 2, 2+a, 2+2a, 2+3a,...} where a=offset. 
            The offset should be less than the num_subsets
        
        Example
        -------
        >>> sampler=Sampler.staggered(21,4)
        >>> print(sampler.get_samples(5))
        >>> for _ in range(15):
        >>>    print(sampler.next())
        >>> print(sampler.get_samples(5))
        
        [ 0  4  8 12 16]
        0
        4
        8
        12
        16
        20
        1
        5
        9
        13
        17
        2
        6
        10
        14
        [ 0  4  8 12 16]
        """
        if offset>=num_subsets:
            raise(ValueError('The offset should be less than the number of subsets'))
        indices=list(range(num_subsets))
        order=[]
        [order.extend(indices[i::offset]) for i in range(offset)]      
        sampler=Sampler(num_subsets, sampling_type='staggered', order=order)
        return sampler 
    
    

    @staticmethod
    def randomWithReplacement(num_subsets, prob=None, seed=None):
        """
        Function that takes a number of subsets and returns a sampler which outputs from a list of integers {0, 1, …, S-1} with S=num_subsets with given probability and with replacement. 

        num_subsets: int
            The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. 

        prob: list of floats of length num_subsets that sum to 1. default=None
            This is the probability for each integer to be called by next. If None, then the integers will be sampled uniformly. 

        seed:int, default=None
            Random seed for the random number generator.  If set to None, the seed will be set using the current time.

        
        Example
        -------
    

        >>> sampler=Sampler.randomWithReplacement(5)
        >>> print(sampler.get_samples(10))

        [3 4 0 0 2 3 3 2 2 1]

        Example
        ------- 

        >>> sampler=Sampler.randomWithReplacement(4, [0.7,0.1,0.1,0.1])
        >>> print(sampler.get_samples(10))

        [0 1 3 0 0 3 0 0 0 0]
        """
          
        if prob==None:
            prob = [1/num_subsets] *num_subsets
        sampler=Sampler(num_subsets, sampling_type='random_with_replacement', prob=prob, seed=seed)
        return sampler 
    
    @staticmethod
    def randomWithoutReplacement(num_subsets, seed=None, shuffle=True):
        
        """
        Function that takes a number of subsets and returns a sampler which outputs from a list of integers {0, 1, …, S-1} with S=num_subsets uniformly randomly without replacement.
        

        num_subsets: int
            The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. 

        seed:int, default=None
            Random seed for the  random number generator.  If set to None, the seed will be set using the current time. 

        shuffle:boolean, default=True
            If True, there is a random shuffle after all the integers have been seen once, if false the same random order each time the data is sampled is used.
        Example
        -------
        >>> sampler=Sampler.randomWithoutReplacement(11)
        >>> print(sampler.get_samples(12))
        [ 1  7  6  3  2  8  9  5  4 10  0  4]
        """
        
        order=list(range(num_subsets))
        sampler=Sampler(num_subsets, sampling_type='random_without_replacement', order=order, shuffle=shuffle, seed=seed)
        return sampler 


    def __init__(self, num_subsets, sampling_type, shuffle=False, order=None, prob=None, seed=None):
        """
        This method is the internal init for the sampler method. Most users should call the static methods e.g. Sampler.sequential or Sampler.staggered. 

        Parameters
        ----------
        num_subsets: int
            The sampler will select from a list of integers {0, 1, …, S-1} with S=num_subsets. 
        
        sampling_type:str
            The sampling type used. 

        order: list of integers
            The list of integers the method selects from using next. 
        
        shuffle= bool, default=False
            If True, after each num_subsets calls of next, the sampling order is shuffled randomly. 

        prob: list of floats of length num_subsets that sum to 1. 
            For random sampling with replacement, this is the probability for each integer to be called by next. 

        seed:int, default=None
            Random seed for the methods that use a random number generator. If set to None, the seed will be set using the current time.  
        """
        self.type=sampling_type
        self.num_subsets=num_subsets
        if seed !=None:
            self.seed=seed
        else:
            self.seed=int(time.time())
        self.generator=np.random.RandomState(self.seed)
        self.order=order
        self.initial_order=order
        if order!=None:
            self.iterator=self._next_order
        self.prob=prob
        if prob!=None:
            self.iterator=self._next_prob
        self.shuffle=shuffle
        self.last_subset=self.num_subsets-1
        


    
    def _next_order(self):
        """ 
        The user should call sampler.next() or next(sampler) rather than use this function. 

        A function of the sampler that selects from a list of integers {0, 1, …, S-1}, with S=num_subsets, the next sample according to the type of sampling.

        This function is used by samplers that sample without replacement. 
         
        """
      #  print(self.last_subset)
        if self.shuffle==True and self.last_subset==self.num_subsets-1:
                self.order=self.generator.permutation(self.order)
                #print(self.order)
        self.last_subset= (self.last_subset+1)%self.num_subsets
        return(self.order[self.last_subset])
    
    def _next_prob(self):
        """ 
        The user should call sampler.next() or next(sampler) rather than use this function. 

        A function of the sampler that selects from a list of integers {0, 1, …, S-1}, with S=num_subsets, the next sample according to the type of sampling.

        This function us used by samplers that select from a list of integers {0, 1, …, S-1}, with S=num_subsets, randomly with replacement. 
         
        """
        return int(self.generator.choice(self.num_subsets, 1, p=self.prob))

    def next(self):
        """ A function of the sampler that selects from a list of integers {0, 1, …, S-1}, with S=num_subsets, the next sample according to the type of sampling. """

        return (self.iterator())

    def __next__(self):
        """ 
        A function of the sampler that selects from a list of integers {0, 1, …, S-1}, with S=num_subsets, the next sample according to the type of sampling. 

        Allows the user to call next(sampler), to get the same result as sampler.next()"""
        return(self.next())

   
    def get_samples(self,  num_samples=20):
        """
        Function that takes an integer, num_samples, and returns the first num_samples as a numpy array.

        num_samples: int, default=20
            The number of samples to return. 

        Example
        -------

        >>> sampler=Sampler.randomWithReplacement(5)
        >>> print(sampler.get_samples())
        [2 4 2 4 1 3 2 2 1 2 4 4 2 3 2 1 0 4 2 3]

        """
        save_generator=self.generator
        save_last_subset=self.last_subset
        self.last_subset=self.num_subsets-1
        save_order=self.order
        self.order=self.initial_order
        self.generator=np.random.RandomState(self.seed)
        output=[self.next() for _ in range(num_samples)]
        self.generator=save_generator
        self.order=save_order
        self.last_subset=save_last_subset
        return(np.array(output))

