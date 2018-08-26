import numpy
from layers import convolution_2d
from layers import pooling_2d
from layers import flatten
from layers import dense
from loss_functions import log_loss
from activations import sigmoid



numpy.random.seed(0)
X = numpy.random.rand(1,3,3)
Y = numpy.array((0,1,1,0,1,0,0,1,1,0))
print(numpy.shape(X))
epochs=1
for i in range(epochs):
    # Define the layers of the model
    conv_1=convolution_2d.layer(1,filter_d=2,stride=1,padding=0)
    output=conv_1.evaluate(X)
    print(numpy.shape(output))

    dlds=log_loss.evaluate(0,output,derivative=1)

    conv_bp_1=convolution_2d.layer(len(dlds),weights=dlds,filter_d=2,stride=1,padding=0)
    final=conv_bp_1.evaluate(X)

    print(final)

    '''
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

    output=flatten.evaluate(output)

    dense_1=dense.layer(10)
    output=dense_1.evaluate(output)
    print(numpy.shape(output))

    #output=sigmoid.evaluate(output)
    output=((1.0)*(output-numpy.min(output))/(numpy.max(output)-numpy.min(output)))
    print(output)

    prediction=numpy.max(output)

    print()
    '''
