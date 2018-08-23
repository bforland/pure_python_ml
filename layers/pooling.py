'''
First attempt at pooling code.

Blake Forland @ RRCC DataLab

'''

import numpy as np
from numpy import random
import logging
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('RuntimeWarning')
    function_raising_warning()

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")

filter_size_1 = 3.0 # Set the step size
filter_size_2 = 2.0 # Set the step size
stride_1 = 2.0
stride_2 = 2.0
dim_1 = 6.0 # Set the dim of the square matrix
dim_2 = 6.0
vals = [] # Empty list to catch the values

logging.info("Data dimension: "+str(dim_1)+" x "+str(dim_2))
logging.info("Filter dimentions: "+str(filter_size_1)+" x "+str(filter_size_2))
logging.info("Stride lengths: "+str(stride_1)+", "+str(stride_2))

# Creating pool filter dimensions and checking to make sure they work
pool_dim_1 = (dim_1-(filter_size_1-stride_1))/stride_1
pool_dim_2 = (dim_2-(filter_size_2-stride_2))/stride_2
dim_check_1 = (np.round(pool_dim_1)-np.round(pool_dim_1,1))
dim_check_2 = (np.round(pool_dim_2)-np.round(pool_dim_2,1))

# Checking the dimension TypeError

try:
    x=1/dim_check_1
    x=1/dim_check_2


    # Parameters have to be integers
    filter_size_1,filter_size_2,stride_1,stride_2,dim_1,dim_2=np.int(filter_size_1),np.int(filter_size_2),np.int(stride_1),np.int(stride_2),np.int(dim_1),np.int(dim_2)

    # Create a random square matrix
    A = np.floor(10.0 * (np.random.rand(dim_1,dim_2)))

    # Check to see A
    logging.info("Data:")
    for r in A:
        logging.info("        "+str(r))

    # Set the start index for i
    i = 0

    # Loop over all the entries of the matrix, modified by the pool filter
    while i < (dim_1-(filter_size_1-stride_1)):

        # Set (reset) the start index for j
        j = 0
        while j < (dim_2-(filter_size_2-stride_2)):

            # Pull the section that we want to
            # pool, find the max and append it
            # to the list of values.
            vals.append(np.max(A[i:i+filter_size_1,j:j+filter_size_2]))

            # Index up by the stride length
            j += stride_2
        i += stride_1
    # Reshape and print the new "Pool" matrix
    vals=np.reshape(np.array(vals),(np.int(pool_dim_1),np.int(pool_dim_2)))

    logging.info("MaxPooled output:")
    for r in vals:
        logging.info("        "+str(r))
except RuntimeWarning:
    logging.info("Checking that filter, stride and data dims are compatable.")
    if dim_check_1 != 0:
        logging.info("Dimension 1: FILTER and STRIDE size don't work with DATA dimensions")
        logging.info("("+str(dim_1)+"-("+str(filter_size_1)+"-"+str(stride_1)+"))/"+str(stride_1)+" = "+str(pool_dim_1))
        logging.info("Above must be integer")
    if dim_check_2 != 0:
        logging.info("Dimension 2: FILTER and STRIDE size don't work with DATA dimensions")
        logging.info("("+str(dim_1)+"-("+str(filter_size_2)+"-"+str(stride_2)+"))/"+str(stride_2)+" = "+str(pool_dim_2))
        logging.info("Above must be integer")

except ValueError:
    logging.info("Checking that filter, stride and data dims are compatable.")
    if dim_check_1 != 0:
        logging.info("Dimension 1: FILTER and STRIDE size don't work with DATA dimensions")
        logging.info("("+str(dim_1)+"-("+str(filter_size_1)+"-"+str(stride_1)+"))/"+str(stride_1)+" = "+str(pool_dim_1))
        logging.info("Above must be integer")
    if dim_check_2 != 0:
        logging.info("Dimension 2: FILTER and STRIDE size don't work with DATA dimensions")
        logging.info("("+str(dim_1)+"-("+str(filter_size_2)+"-"+str(stride_2)+"))/"+str(stride_2)+" = "+str(pool_dim_2))
        logging.info("Above must be integer")
