import numpy

def evaluate(y, y_pred):
    return -(y * numpy.log(y_pred) + (1. - y) * numpy.log(1. - y_pred))
