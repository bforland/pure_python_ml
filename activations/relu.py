import numpy

def relu(x, alpha=0.0):
    if alpha == 0.0:
        return 0.5 * (x + numpy.abs(x))
    else:
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * numpy.abs(x)
