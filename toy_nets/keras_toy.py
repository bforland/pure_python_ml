from sklearn.datasets import load_breast_cancer
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy
from keras.callbacks import History
history = History()

model = Sequential()
model.add(Dense(units = 100, activation = 'sigmoid', input_dim = 30))
model.add(Dense(units = 1, activation='sigmoid'))
model.summary()
model.compile(loss = keras.losses.binary_crossentropy,optimizer = 'rmsprop',metrics=['accuracy'])

#data = load_breast_cancer()
#X = data.data
#Y = data.target

sig = (numpy.random.randn(569,30)+.2)
bkg = (numpy.random.randn(569,30)-.2)
sig_labels = numpy.ones(569)
bkg_labels = numpy.zeros(569)
X = numpy.append(sig,bkg)
X=numpy.reshape(X,(1138,30))
Y = numpy.append(sig_labels,bkg_labels)

s = numpy.arange(len(X))

numpy.random.shuffle(s)

X = X[s]
Y = Y[s]


x_train=X[:1000]
y_train=Y[:1000]
x_test=X[1000:]
y_test=Y[1000:]

history = model.fit(x_train, y_train, epochs = 15, batch_size = 32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size = 128)
test_auc=roc_auc_score(y_test, classes)
print(test_auc)
plt.scatter(numpy.arange(len(history.history['loss'])),history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
sh,_,_=plt.hist(numpy.reshape(classes,(138))[numpy.where(y_test==1)],bins=10,alpha=.5,color='red')
bh,_,_=plt.hist(numpy.reshape(classes,(138))[numpy.where(y_test==0)],bins=10,alpha=.5,color='blue')
plt.xlabel('Network Response')
plt.ylabel('Density')
plt.show()
