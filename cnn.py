import numpy

numpy.random.seed(0)
X = numpy.floor(10.0 * (numpy.random.rand(10,6,6)))
print(numpy.shape(X))
