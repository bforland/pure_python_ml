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

data = load_breast_cancer()
X = data.data
Y = data.target

x_train=X[:500]
y_train=Y[:500]
x_test=X[500:569]
y_test=Y[500:569]

history = model.fit(x_train, y_train, epochs = 15, batch_size = 32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size = 128)
test_auc=roc_auc_score(y_test, classes)
print(test_auc)
plt.scatter(numpy.arange(len(history.history['loss'])),history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
sh,_,_=plt.hist(numpy.reshape(classes,(69))[numpy.where(y_test==1)],bins=10,alpha=.5,color='red')
bh,_,_=plt.hist(numpy.reshape(classes,(69))[numpy.where(y_test==0)],bins=10,alpha=.5,color='blue')
plt.xlabel('Network Response')
plt.ylabel('Density')
plt.show()
