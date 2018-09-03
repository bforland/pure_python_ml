import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import roc_auc_score
import scipy.special

def sigmoid(x):
    return scipy.special.expit(x)

def d_sigmoid(x):
    return scipy.special.expit(x)*(1-scipy.special.expit(x))

def CrossEntropy(y,p,eps=1e-15):
    return -(y*numpy.log(p+eps)+(1.-y)*numpy.log(1.+eps-p))

def d_CrossEntropy(y,p,eps=1e-15):
    return (eps+p-y-2.*eps*y)/(eps+numpy.power(eps,2)+p-numpy.power(p,2))

def RMS(y,p):
    return .5*(y - p)**2

def d_RMS(y,p):
    return (y - p)

def forward(X,Y,weights_1,weights_2):

    # Calculate the Sum of the inputs an the hidden weights
    hidden_inputs = numpy.dot(weights_1, numpy.transpose(X))

    # calculate the signals emerging from hidden layer
    hidden_outputs = scipy.special.expit(hidden_inputs)

    # Calculate the Sum of the hidden layer an the output weights
    final_inputs = numpy.dot(weights_2, hidden_outputs)

    # calcuate the signals emerging from final output layer
    final_outputs  = scipy.special.expit(final_inputs)

    return final_outputs,final_inputs,hidden_outputs,hidden_inputs

def backwards(X,Y,final_inputs,final_outputs,hidden_inputs,hidden_outputs,weights_2,weights_1):

    output_errors = numpy.multiply(d_RMS(Y,final_outputs.T),numpy.transpose(d_sigmoid(final_inputs)))
    #print(numpy.mean(output_errors))
    hidden_errors = numpy.multiply(numpy.dot(output_errors,weights_2),numpy.transpose(d_sigmoid(hidden_inputs)))

    weights_2 = weights_2 + (lr * numpy.dot(hidden_outputs,output_errors).T)

    weights_1 = weights_1 + (lr * numpy.dot(hidden_errors.T,X))

    return weights_1,weights_2

events = 500
epochs = 15
lr = .05
batch_size = 16
n_batches = int(numpy.ceil(events / batch_size))
print(n_batches)

data = load_breast_cancer()
my_data = data.data
targets = data.target

X = my_data[:events]
Y = targets[:events]
Y = numpy.reshape(Y,(events,1))

loss = []

layer_1 = 100
layer_2 = 1

weights_1 = numpy.random.normal(0.0, numpy.sqrt(2. / (30 + layer_1)), (layer_1, 30))
weights_2 = numpy.random.normal(0.0, numpy.sqrt(2. / (layer_1 + layer_2)), (layer_2,layer_1))

for i in range(epochs):

    outputs = numpy.empty(0)

    for b in range(n_batches):

        b_start = b * batch_size

        b_end = (b + 1) * batch_size

        X_batch = X[b_start:b_end]

        X_batch = 2. * (X_batch - numpy.min(X_batch))/(numpy.max(X_batch)-numpy.min(X_batch)) + 1

        final_outputs,final_inputs,hidden_outputs,hidden_inputs = forward(X_batch,Y[b_start:b_end],weights_1,weights_2)

        weights_1,weights_2 = backwards(X_batch,Y[b_start:b_end],final_inputs,final_outputs,hidden_inputs,hidden_outputs,weights_2,weights_1)

        outputs = numpy.append(outputs,final_outputs)

    #outputs = numpy.reshape(outputs,(len(outputs),1))
    #print(numpy.mean(RMS(numpy.reshape(Y[:len(outputs)],(len(outputs),)),outputs.T)))
    loss.append(numpy.mean(RMS(numpy.reshape(Y[:len(outputs)],(len(outputs),)),outputs.T)))

Y = numpy.reshape(Y[:len(outputs)],(len(outputs),))
print(numpy.mean(outputs[numpy.where(Y==0)]))
print(outputs[numpy.where(Y==0)][:10])
print(numpy.mean(outputs[numpy.where(Y==1)]))
print(outputs[numpy.where(Y==1)][:10])
test_auc=roc_auc_score(Y, outputs)
print(test_auc)
plt.scatter(numpy.arange(len(loss)),loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
sh,_,_=plt.hist(outputs[numpy.where(Y==1)],bins=10,alpha=.5,color='red')
bh,_,_=plt.hist(outputs[numpy.where(Y==0)],bins=10,alpha=.5,color='blue')
plt.xlabel('Network Response')
plt.ylabel('Density')
plt.show()
