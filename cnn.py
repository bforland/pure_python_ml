import numpy
from layers import convolution_2d
from layers import pooling_2d
from layers import flatten
from layers import dense
from loss_functions import log_loss
from activations import sigmoid

model = numpy.load("model.npy")

model.info()

output = sgd(model)

outputs = []

for e in epochs:
    
    for l,layer in enumerate(layers):

        X = layer.evaluate(X)

        outputs.append(X)

#    conv_2=convolution_2d.layer(10,filter_d=3,stride=1,padding=0)

#    act_1=sigmoid.evaluate(output)

#    pool_2=pooling_2d.layer(filter_size=3)

#    conv_3=convolution_2d.layer(10,filter_d=3,stride=1,padding=0)

#    output=sigmoid.evaluate(output)

#    pool_3=pooling_2d.layer(filter_size=3)

#    output=flatten.evaluate(output)

#    dense_1=dense.layer(100)

#    output=sigmoid.evaluate(output)

#    output=flatten.evaluate(output)

#    dense_2=dense.layer(10)

#    print(output)

#    prediction=numpy.max(output)

#    print()
#    '''
