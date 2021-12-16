"""
functional apis lets users to:
separate layers from model,
have user-defined connections between layers
Make Functions out of units of layers"""

from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_data(pl, X, y):  # plot the data on a figure
    pl.plot(X[y == 0, 0], X[y == 0, 1], 'ob', alpha=0.5)  # plot class where y==0 in color blue as 'o'
    pl.plot(X[y == 1, 0], X[y == 1, 1], 'xr', alpha=0.5)  # plot class where y==1 in color red as 'x'
    pl.legend(['0', '1'])
    return pl


def plot_decision_boundary(model, X, y):  # Common function that draws the decision boundaries
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    c = model.predict(ab)  # make prediction with the model
    Z = c.reshape(aa.shape)  # reshape the output so contourf can plot it

    plt.figure(figsize=(12, 8))
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)  # plot contour rather than single line to see prediction confidence
    plot_data(plt, X, y)  # plot the moons of data

    return plt


# Generate some data blobs.  Data will be either 0 or 1 when 2 is number of centers.
# X is a [number of samples, 2] sized array. X[sample] contains its x,y position of the sample in the space
# ex: X[1] = [1.342, -2.3], X[2] = [-4.342, 2.12]
# y is a [number of samples] sized array. y[sample] contains the class index (ie. 0 or 1 when there are 2 centers)
# ex: y[1] = 0 , y[1] = 1
X, y = make_circles(n_samples=1000, factor=.6, noise=0.1, random_state=42)
# factor-> scale factor between inner and outer circle
# noise-> std deviation of gaussian noise added to data

pl = plot_data(plt, X, y)
pl.show()

"""
there is more complex cluster, the data is not so easily linearly separable,
hence hidden layers are required to classify data, but here we are seeing what happens if we dont add hidden layers"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# test_size-> fraction of dataset that is put in the test set i,e, X_test & y_test
# random_state -> to have split when code is run multiple times

from keras.models import Model
from keras.layers import Dense, Input  # lets you to define an input layer
from keras.optimizers import Adam  # performs back-propagation

"""
General flow in functional api model
creates layers and connect layers
define model by specifying input to output,
compile the model,
train the model with the training data,
evaluate the performance of the model against the testing/validation data
"""

ip = Input(shape=(2,))  # define input layer
x1 = Dense(4, activation="tanh", name="Hidden-1")(ip)  # define ip layer and connect it to 1st hidden layer
x2 = Dense(4, activation="tanh", name="Hidden-2")(x1)  # define 1st hidden layer and connect it to 2nd one
op = Dense(1, activation="sigmoid", name="Output_layer")(x2)  # define 2nd hidden layer and connect it to op layer
model = Model(inputs=ip, outputs=op)  # create model by specifying input and output layers
"""
Many a times as models become complex it's difficult to visualize what's happening.
So Summary will help by giving info regarding total params- trainable/non-trainable.
This gives idea of complexity of model. When the #params increase, more time is required to train the model."""
model.summary()

"""Adam optimizer is used to minimize the loss, which is how often the model incorrectly predicts the class."""
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])  # define model's learning process
# lr -> learning rate
# 'binary_crossentropy' -> function to calculate the loss,
# metrics=['accuracy'] -> optimize accuracy

"""For documentation, we can generate a png of model as well"""
from keras.utils import plot_model

plot_model(model, to_file="2_3_comp_clust_2_hid_summary.png", show_shapes=True, show_layer_names=True)

"""
Keras lets us add callback(pre/user-defined) to gain various capapbilities during runtime.
Ex: control/obtain additional info etc."""
from keras.callbacks import EarlyStopping

my_callbacks = [EarlyStopping(monitor='val_acc', patience=5, mode='max')]  # stop early callback configuration
# monitor='val_acc' -> based on validation accuracy
# patience=5 -> if no change in accuracy for 5 epochs stop training
# mode='max' -> quantity should be increasing
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks, validation_data=(X_test, y_test))
# Adjust the weight and bias to minimize a loss.
# epochs=100 -> 100 runs through the training data, and on each run the optimizer will adjust the weights and biases to
# minimize the loss and increase the accuracy.
# callback -> callback function to be called
eval_result = model.evaluate(X_test, y_test)  # Evaluate model -> how well model classifies data
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])  # Print test accuracy
plot_decision_boundary(model, X, y).show()  # Plot the decision boundary
