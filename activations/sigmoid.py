import numpy

def evaluate(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+numpy.exp(-x))
