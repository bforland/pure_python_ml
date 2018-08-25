import numpy
from layers import convolution_2d
from layers import pooling_2d
from layers import flatten
from layers import dense
from activations import sigmoid



numpy.random.seed(0)
X = numpy.random.rand(10,25,25)
print(numpy.shape(X))

# Define the layers of the model
conv_1=convolution_2d.layer(10,filter_d=11,stride=1,padding=0)
output=conv_1.evaluate(X)
print(numpy.shape(output))

output=sigmoid.evaluate(output)

pool_1=pooling_2d.layer()
output=pool_1.evaluate(output)
print(numpy.shape(output))

conv_2=convolution_2d.layer(10,filter_d=3,stride=1,padding=0)
output=conv_2.evaluate(output)
print(numpy.shape(output))

output=sigmoid.evaluate(output)

pool_2=pooling_2d.layer(filter_size=3)
output=pool_2.evaluate(output)
print(numpy.shape(output))

conv_3=convolution_2d.layer(10,filter_d=3,stride=1,padding=0)
output=conv_3.evaluate(output)
print(numpy.shape(output))

output=sigmoid.evaluate(output)

pool_3=pooling_2d.layer(filter_size=3)
output=pool_3.evaluate(output)
print(numpy.shape(output))

output=flatten.evaluate(output)
print(numpy.shape(output))

dense_1=dense.layer(100)
output=dense_1.evaluate(output)
print(numpy.shape(output))

output=sigmoid.evaluate(output)

dense_1=dense.layer(2)
output=dense_1.evaluate(output)
print(numpy.shape(output))

output=sigmoid.evaluate(output)

print(output)
