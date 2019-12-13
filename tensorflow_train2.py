import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
from tensorflow.keras.callbacks import TensorBoard
import time

"""
Attributions:
 taken from tensorflow.org as examples for creation of this model
https://www.tensorflow.org/tutorials/
https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/?completed=/loading-custom-data-deep-learning-python-tensorflow-keras/
"""

NAME = f"Utensil-image-recognition-64x3-{int(time.time())}"

tensorboard = TensorBoard(log_dir=f"logs\\{NAME}")


data = np.load('data_features.npy')

label = np.load('labels_list.npy')

data = data/255.0

# generate 2d classification dataset
data, label = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
# encode output variable
label = to_categorical(label)

print(data.shape)


"""" splitting data """
# split into train and test
train_size = 80
trainX = data[:train_size, :]
trainy = label[:train_size]

testX = data[train_size:, :]
testy = label[train_size:]
def create_model():

    """ create the model """
    # define model
    new_model = Sequential()
    new_model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    new_model.add(Flatten())
    new_model.add(Dense(3, activation='softmax'))

    # create optimizer
    opt = Adam(lr=0.01)

    # compile model
    new_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return new_model


# create the model
model = create_model()

# fit model
model.fit(trainX, trainy, batch_size=32, epochs=100, verbose=2, callbacks=[tensorboard])


# evaluate the model
loss, acc = model.evaluate(trainX, trainy, verbose=2)
# _, train_acc = model.evaluate(trainX, trainy, verbose=0)
# _, test_acc = model.evaluate(testX, testy, verbose=0)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
print('Trained model, accuracy: {:5.2f}%'.format(100*acc))


#model.save('image_recognition.h5')
#print('saved model')

"""
# plot loss during training
#
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
"""


