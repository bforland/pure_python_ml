import numpy

def evaluate(true_label, predicted, eps=1e-15,derivative=0):
    if derivative == 0:
        if true_label.any() == 1:
            return -numpy.log(predicted[numpy.where(true_label==1)])
        else:
            return -numpy.log(1 - predicted[numpy.where(true_label==0)])
    else:
        return 1.0/predicted
