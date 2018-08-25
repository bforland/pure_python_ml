import numpy

def evaluate(data):
    return numpy.reshape(data,(numpy.product(numpy.shape(data))))
