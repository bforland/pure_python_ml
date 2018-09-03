import numpy
from layers import convolution_2d
from layers import pooling_2d
from layers import flatten
from layers import dense
from loss_functions import log_loss
from activations import sigmoid

'''
# Forward feed on the network ---------->
def forward(X, weights_1, weights_2):

    # X ---> X * W0 --sig--> z1 ---> z1 * W2 --sig--> z2

    # Calculate the Sum of the inputs an the hidden weights
    hidden_inputs = numpy.dot(weights_1, numpy.transpose(X))
    diprint("Hidden inputs: " + str(hidden_inputs), User_check)

    # Calculate the signals emerging from hidden layer
    hidden_outputs = activation.sigmoid(hidden_inputs, False)
    diprint("Hidden outputs: " + str(hidden_outputs), User_check)

    # Calculate the Sum of the hidden layer an the output weights
    final_inputs = numpy.dot(weights_2, hidden_outputs)
    # Calcuate the signals emerging from final output layer
    final_outputs = activation.sigmoid(final_inputs, False)

    return final_outputs, final_inputs, hidden_outputs, hidden_inputs

'''

numpy.random.seed(0)
X = numpy.random.rand(1,3,3)
Y = numpy.array((0,1,1,0,1,0,0,1,1,0))
print(numpy.shape(X))
epochs=1

layers =[]

layers.append(dense.layer(100))

layers.append(sigmoid)

layers.append(dense.layer(1))

layers.append(sigmoid)

outputs = []

for layer in layers:

    X = layer.evaluate(X)

    outputs.append(X)

error = loss_function(outputs[len(layers -1)])
