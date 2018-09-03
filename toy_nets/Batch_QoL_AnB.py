'''
Baseline feedforward neural network for teaching purpose.
'''
#--- THIS INDICATES A TODO ---#

#--- We should remove numpy ---#
import numpy
from numpy import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

import logging
import time

logging.basicConfig(level=logging.INFO,format="[%(asctime)s %(levelname)s] %(message)s")


# diagonastic printer (use: diprint(STRING,User_check))
def diprint(S, bol):
    if bol == "Y" or bol == "y":
        print(S)
    elif bol == 'n' or bol == None:
        pass


# collection of Activation functions.
class activation:
    def sigmoid(x, D):
        if D is False:
            return 1. / (1. + numpy.exp(-x))
        if D is True:
            return((1.- activation.sigmoid(x,False)) * activation.sigmoid(x,False))

    def RMS(y, p, D):
        if D is False:
            return .5*(y - p)**2
        if D is True:
            return (y - p)

# normalization styles]
class normalization:
  def minmax(self,data,b=1.,a=-1.):
    return (b - a) * (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data)) - a
  def mean(self,data,b=1.,a=-1.):
    return (b - a) * (data - numpy.mean(data)) / (numpy.max(data) - numpy.min(data)) - a
  def standard(self,data):
    return (data - numpy.mean(data))/numpy.std(data)

# Performance of a net based off an desired AUC score
net_performance = lambda time,auc,loss: (time+1.)**(numpy.e**((.95-auc)*10.+loss))

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


# Backpropagation <----------
def backwards(X, Y, final_outputs, final_inputs, hidden_outputs, hidden_inputs, weights_2, weights_1):

    diprint("Shape of X: " + str(numpy.shape(X)), User_check)
    diprint("Shape of weights_1: " + str(numpy.shape(weights_1)), User_check)
    diprint("Shape of hidden_outputs: " + str(numpy.shape(hidden_outputs)), User_check)
    diprint("Shape of weights_2: " + str(numpy.shape(weights_2)), User_check)
    diprint("Shape of final_outputs: " + str(numpy.shape(final_outputs)), User_check)
    diprint("Shape of Y: " + str(numpy.shape(Y)), User_check)

    # Differences between truth and found.
    output_errors = activation.RMS(Y, final_outputs.T, True)

    # Chain rule to get the output delta.
    output_delta = numpy.multiply(output_errors,
            numpy.transpose(activation.sigmoid(final_inputs, True)))

    # Backpropagation to hidden errors.
    hidden_errors = numpy.dot(output_delta, weights_2)

    # Chain rule to get hidden delta.
    hidden_delta = numpy.multiply(hidden_errors,
            numpy.transpose(activation.sigmoid(hidden_inputs, True)))

    # Update the weights.
    weights_1 += lr * numpy.dot(hidden_delta.T, X)
    weights_2 += lr * numpy.dot(hidden_outputs, output_delta).T

    return weights_1, weights_2 # Return the updated weights.

### USER MENU ###

# Ask user if they would like to print
# all the shapes.
def_params = input("Run defaults? ")

if def_params != "y" and def_params != "Y":
  User_check = input("Print diagnostics (Y/n)? ")
  make_plots = input("Make plots (Y/n)? ")
  epochs = int(input("Epochs: "))
  lr = float(input("Learning Rate: "))
  batch_size = int(input("Batch size: "))

  early_stop = input("Early stopping (Y/n)?  ")
  if early_stop == "y" or early_stop == "Y":
    early_stop = float(input("% Loss change threshold: "))
    early_stop = early_stop / 100.
    tolerance = int(input("Epochs tolerance: "))
  else:
    early_stop = 0.0
    tolerance = 0.0
else:
  User_check = "n"
  make_plots = "y"
  epochs = 20
  lr = .1
  batch_size = 32
  early_stop = 0.0
  tolerance = 0.0
#################

# Load the data.
data = load_breast_cancer()
my_data = data.data
targets = data.target

# Set the number of events.
events = 500

# Calculate the number of batches.
n_batches = int(events / batch_size)

# Build the data and the targets
X = my_data[:events]
Y = targets[:events]

#--- could we use a transpose here? --#
Y = numpy.reshape(Y, (events, 1))

diprint(numpy.shape(X), User_check)
diprint(numpy.shape(Y), User_check)

layer_1 = 100 # Layer_1 number of nodes.
layer_2 = 1 # Layer_2 number of nodes.

# Build the weights for the layers (random to start)
weights_1 = numpy.random.normal(0.0, numpy.sqrt(2. / (30 + layer_1)), (layer_1, 30))
weights_2 = numpy.random.normal(0.0, numpy.sqrt(2. / (layer_1 + layer_2)), (layer_2, layer_1))

# List to contain loss values
metrics = {"loss":[],"acc":[],"auc":[]}

es_count = 0
loss_temp = 0

t1 = time.time()

# Loop over the epochs.
for e in range(epochs):

    outputs = numpy.empty(0)

    # Loop over the batches.
    for b in range(n_batches):

        diprint("We are on batch " + str(b), User_check)
        diprint("The batch size is " + str(batch_size), User_check)

        b_start = b * batch_size # Start index for the batch.
        b_end = (b + 1) * batch_size # End index for the batch.

        X_batch = X[b_start:b_end] # Array of batch size from training data.

        # Normalize the batch
        X_batch = normalization.standard(X_batch)

        # Send the data through the network forward -->
        final_outputs, final_inputs, hidden_outputs, hidden_inputs = forward(X_batch, weights_1, weights_2)

        # Send the errors backward through the network <--
        weights_1, weights_2 = backwards(X_batch,
                Y[b_start:b_end], final_outputs, final_inputs,
                hidden_outputs, hidden_inputs, weights_2, weights_1)

        outputs = numpy.append(outputs, final_outputs)

    outputs = numpy.reshape(outputs, (len(outputs), 1))

    metrics["loss"].append(numpy.round(numpy.mean(activation.RMS(Y[:len(outputs)], outputs, False)),4))
    metrics["acc"].append(numpy.round(numpy.mean(numpy.equal(Y[:len(outputs)],numpy.round(outputs))),4))
    metrics["auc"].append(numpy.round(roc_auc_score(Y[:len(outputs)], outputs),4))
    if(e%(epochs*.1)==0):
      logging.info(" Epoch: "+str(e+1)+" Loss: "+str(metrics["loss"][e])+" Acc: "+str(metrics["acc"][e])+" AUC: "+str(metrics["auc"][e]))
    if e > 0:
      if numpy.abs(1. - (metrics["loss"][e]/metrics["loss"][e-1])) < early_stop:
        if es_count < tolerance:
          es_count += 1
          loss_temp += numpy.abs(1.-(metrics["loss"][e]/metrics["loss"][e-1]))
        else:
          logging.info(" Epoch: "+str(e)+" Stopping Early!")
          logging.info(" Averge loss change of "+str(numpy.round(100.*loss_temp/es_count,2))+"% over "+str(es_count)+" consecutive epochs is below required threshold of "+str(early_stop*100))
          break
      else:
        es_count = 0
        loss_temp = 0
logging.info(" Final - Loss: "+str(metrics["loss"][len(metrics["loss"])-1])+" Acc: "+str(metrics["acc"][len(metrics["acc"])-1])+" AUC: "+str(metrics["auc"][len(metrics["auc"])-1]))
logging.info(" Run time: "+str(numpy.round(time.time()-t1,2)))
logging.info(" Performance for .95 AUC Score: "+str(net_performance(time.time()-t1,metrics["auc"][len(metrics["auc"])-1],metrics["loss"][1]-metrics["loss"][len(metrics["loss"])-1])))
diprint(numpy.shape(weights_1), User_check)
diprint(numpy.shape(weights_2), User_check)

if make_plots == "y" or make_plots == "Y":
  fig, ax1 = plt.subplots()
  ax1.plot(numpy.arange(len(metrics["loss"])-1),normalization.minmax(metrics["loss"][1:],1.,0.),color='blue')
  ax1.set_xlabel('Epochs')

  ax1.set_ylabel("Loss (Normed)", color='b')
  ax1.tick_params('y', colors='b')

  ax2 = ax1.twinx()

  ax2.plot(numpy.arange(len(metrics["acc"])-1),metrics["acc"][1:],color='green')
  ax2.set_ylabel('Acc', color='g')
  ax2.tick_params('y', colors='g')
  #ax2.set_ylim(0.,1.)

  fig.tight_layout()
  plt.show()

  sh,_,_=plt.hist(outputs[numpy.where(Y[:len(outputs)]==1)],bins=10,alpha=.5,color='red')
  bh,_,_=plt.hist(outputs[numpy.where(Y[:len(outputs)]==0)],bins=10,alpha=.5,color='blue')
  plt.xlabel('Network Response')
  plt.ylabel('Density')
  plt.show()
