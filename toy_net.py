import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import auc, roc_auc_score

'''

sig = (numpy.random.randn(10000,10)+1.0)
bkg = (numpy.random.randn(10000,10)-1.0)
sig_labels = numpy.ones(10000)
bkg_labels = numpy.zeros(10000)
X = numpy.append(sig,bkg)
X=numpy.reshape(X,(20000,10))
Y = numpy.append(sig_labels,bkg_labels)

s = numpy.arange(len(X))

numpy.random.shuffle(s)

X = X[s]
Y = Y[s]
'''
digits = load_digits(n_class=2)
X = digits.data
Y = digits.target
print(len(Y))
loss = []

hidden_nodes_1 = 5 # Hidden layer
hidden_nodes_2 = 1 # Output layer
numpy.random.seed(1)
# Initialize weights
weights_1 = numpy.random.rand(len(X[0]),hidden_nodes_1)
weights_2 = numpy.random.rand(hidden_nodes_1,hidden_nodes_2)

def roc_curve_custom(signal,background):

    roc_signal_value=[]
    roc_background_value=[]
    roc_signal_values_sensitive=[]

    for i in range(len(signal)):

        roc_background_value.append(1/(background[i:].sum()/background.sum()))
        roc_signal_value.append(signal[i:].sum()/signal.sum())
        roc_signal_values_sensitive.append((signal[i:].sum()/signal.sum())/(1.5+numpy.sqrt(background[i:].sum())))

    return roc_signal_value,roc_background_value,roc_signal_values_sensitive

def relu(Z):
  Z[numpy.where(Z<0)]=0.0
  return Z

def relu_prime(z):
  z[numpy.where(z>0)]=1.0
  z[numpy.where(z<=0)]=0.0
  return z

def sigmoid(x):
  return 1/(1+numpy.exp(-x))

def D_sigmoid(x):
  return 1-1/(1+numpy.exp(-x))

def logloss(true_label, predicted_prob):
    predicted_prob=numpy.reshape(predicted_prob,(300,))
    for i,l in enumerate(true_label):
        if l == 1:
            predicted_prob[i] = -numpy.log(predicted_prob[i]-1e-15)
        else:
            predicted_prob[i] = -numpy.log(1 - predicted_prob[i]+1e-15)
    return predicted_prob

def backpropagation(output_1,output_2,X,Y):

  output_2_error = (Y-output_2)*D_sigmoid(output_2)

  output_1_error = numpy.transpose(output_2_error).dot(numpy.transpose(weights_2))*numpy.reshape(D_sigmoid(output_1),(300,hidden_nodes_1))

  d2 = output_2_error.dot(numpy.transpose(output_1))

  d1 = numpy.transpose(output_1_error).dot(X)

  return d2,d1

# Optimizer loop
def train(X,Y,weights_1,weights_2):
  epochs = 11
  learning_rate = .001
  for e in range(epochs):
    # Multiply weights and data
    if(e%(epochs*.1)==0):print(e)
    output_1 = numpy.transpose(weights_1).dot(numpy.transpose(X))

    # Activation function
    output_1 = sigmoid(output_1)

    # Multiply weights and data
    output_2 = numpy.transpose(weights_2).dot(output_1)

    # Activation function
    output_2 = sigmoid(output_2)

    loss.append(numpy.mean(logloss(Y,output_2)))

    dW_2,dW_1 = backpropagation(output_1,output_2,X,Y)

    #print(numpy.mean(dW_2),numpy.mean(dW_1))

    weights_1 -= learning_rate * numpy.transpose(dW_1)
    weights_2 -= learning_rate * numpy.transpose(dW_2)

    #print(numpy.mean(0.5 * (output_2 - Y)**2))

  return weights_1,weights_2

def test(X,Y,weights_1,weights_2):

  output_1 = numpy.transpose(weights_1).dot(numpy.transpose(X))

  output_1 = relu(output_1)

  output_2 = numpy.transpose(weights_2).dot(output_1)

  output_2 = sigmoid(output_2)

  return output_2

W_1,W_2 = train(X[:300],Y[:300],weights_1,weights_2)

Y_pred = numpy.reshape(test(X[300:len(Y)],Y[300:len(Y)],W_1,W_2),(60,))
Y=Y[300:len(Y)]
test_auc=roc_auc_score(Y, Y_pred)
print(test_auc)
plt.scatter(numpy.arange(len(loss)),loss)
plt.show()
sh,_,_=plt.hist(Y_pred[numpy.where(Y==1)],range=[0,1])
bh,_,_=plt.hist(Y_pred[numpy.where(Y==0)],range=[0,1])
plt.show()
#s,b,_=roc_curve_custom(sh,bh)
#plt.scatter(s,b)
