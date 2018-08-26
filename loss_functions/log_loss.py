import numpy

def evaluate(true_label, predicted, eps=1e-15):
    p = numpy.clip(predicted, eps, 1 - eps)
    if true_label.any() == 1:
        return -numpy.log(p[numpy.where(true_label==1)])
    else:
        return -numpy.log(1 - p[numpy.where(true_label==0)])
