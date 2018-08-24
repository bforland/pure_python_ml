'''
First attempt at pooling code.

Blake Forland @ RRCC DataLab

'''

import numpy as np
from numpy import random
import logging
import warnings

class layer:
    def __init__(self,filter_size=2,stride=1):
        self.filter=filter_size
        self.stride=stride
        self.pool_dim=0

    def check_dimensions(self,data):
        # Creating pool filter dimensions and checking to make sure they work
        pool_dim = (np.float(np.shape(data[0][0])[0])-(np.float(self.filter)-np.float(self.stride)))/np.float(self.stride)
        dim_check_1 = (np.round(pool_dim)-np.round(pool_dim,1))
        self.pool_dim=int(pool_dim)
    # Checking the dimension TypeError
        return dim_check_1
        
    def pool(self,A):
        vals=[]
        # Set the start index for i
        i = 0

        # Loop over all the entries of the matrix, modified by the pool filter
        while i < (len(A)-(self.filter-self.stride)):

            # Set (reset) the start index for j
            j = 0
            while j < (len(A[0])-(self.filter-self.stride)):

                # Pull the section that we want to
                # pool, find the max and append it
                # to the list of values.
                vals.append(np.max(A[i:i+self.filter,j:j+self.filter]))

                # Index up by the stride length
                j += self.stride
            i += self.stride
        # Reshape and print the new "Pool" matrix
        vals=np.reshape(np.array(vals),(np.int(self.pool_dim),np.int(self.pool_dim)))
        return vals

    def single_filter(self,data):

        pool_X=[]
        for i in range(len(data)):
            pool_X.append(self.pool(data[i]))
        pool_X=np.array(pool_X)

        return pool_X
    def evaluate(self,data):
        if self.check_dimensions(data) == 0:
            maps=[]
            for i in range(len(data)):
                maps.append(self.single_filter(data[i]))
            maps=np.array(maps)
            return maps
