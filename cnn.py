import numpy
from layers import convolution_2d
from layers import pooling_2d
from activations import sigmoid

numpy.random.seed(0)
X = numpy.floor(10.0 * (numpy.random.rand(10,25,25)))
print(numpy.shape(X))


conv_1=convolution_2d.layer(10,filter_d=2,stride=1,padding=0,input_layer=1)
output=conv_1.evaluate(X)
print(numpy.shape(output))

output=sigmoid.evaluate(output)

pool_1=pooling_2d.layer()
output=pool_1.evaluate(output)
print(numpy.shape(output))
