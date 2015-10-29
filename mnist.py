from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint
import keras
from keras.datasets import mnist
from six.moves import range
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from worker import store_to_s3, get_from_s3, get_bucket_items
import tempfile
import time
import StringIO
import requests
import theano
'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).
    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''

max_num_data = 300
temp_file_name = "/tmp/catsvsdogs.hdf5"
batch_size = 128
nb_classes = 10
nb_epoch = 12
data_augmentation = True

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 28, 28
# number of convolutional filters to use at each layer
nb_filters = [32, 32]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = (2, 2)
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = (3, 3)
# the CIFAR10 images are RGB
image_dimensions = 1

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, shapey, shapex)
    X_test = X_test.reshape(X_test.shape[0], 1, shapey, shapex)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters[0], image_dimensions, nb_conv[0], nb_conv[0], border_mode='full'))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters[0], nb_filters[0], nb_conv[0], nb_conv[0]))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[0], nb_pool[0])))
model.add(Dropout(0.25))

model.add(Flatten())
# the image dimensions are the original dimensions divided by any pooling
# each pixel has a number of filters, determined by the last Convolution2D layer
model.add(Dense(nb_filters[0] * (shapex / nb_pool[0]) * (shapey / nb_pool[0]), 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128, nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.best_lost = 9239129


    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        check_for_early_shutdown()
        if logs.get('loss') < self.best_lost:
            print("new best loss %s", logs.get('loss'))
            self.best_lost = logs.get('loss')
            #save_data()

def create_model():
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print("starting model")
    
    model.save_weights(temp_file_name,overwrite=True)
    with open(temp_file_name) as f:
        store_to_s3(str(int(time.time())),'kerasmodels', f.read())
        

def load_model():
    items = get_bucket_items('kerasmodels')
    newest_item = max(map(int,items))
    model_weights = get_from_s3(str(newest_item), 'kerasmodels')
    print("loading model")
    with open('./current.weights', 'w') as f:
        f.write(model_weights)

    model.load_weights("./current.weights")

def check_for_early_shutdown():
    print("resp start")
    try:
        resp = requests.get("http://169.254.169.254/latest/meta-data/spot/termination-time", timeout=0.001)
        if resp.status == 200:
            save_data()
            return 
    except:
        pass
    print("resp end")

def save_data():
    model.save_weights(temp_file_name, overwrite=True)
    with open(temp_file_name, 'r') as f:
        store_to_s3(str(int(time.time())),'kerasmodels', f.read())

def train():
    
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    #checkpointer = ModelCheckpoint(filepath="/Users/quinnjarrell/Desktop/Experiments/keras/saved/", verbose=1, save_best_only=True)
    history = LossHistory()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test),callbacks=[history])
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
import numpy.ma as ma
def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    import matplotlib.pyplot as plt
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im)
    plt.show()

def predict():
    import pylab as pl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    np.set_printoptions(precision=5, suppress=True)
    convout1_f = theano.function([model.get_input(train=False)], convout1.get_output(train=False))
    print(model.predict(X_test[1:5]))
    print(Y_test[1:5])
    #Y_pred = model.predict(X_test)
    # Convert one-hot to index
    #y_pred = np.argmax(Y_pred, axis=1)
    #from sklearn.metrics import classification_report
    #print(classification_report(y_test, y_pred))
    # Visualize convolution result (after activation)
    i = 4600
    # Visualize the first layer of convolutions on an input image
    X = X_test[i:i+1]
    plt.figure()
    plt.title('input')
    nice_imshow(plt.gca(), np.squeeze(X), vmin=0, vmax=1, cmap=cm.binary)
    C1 = convout1_f(X)
    C1 = np.squeeze(C1)
    print("C1 shape : ", C1.shape)

    plt.figure(figsize=(15, 15))
    plt.suptitle('convout1')
    nice_imshow(plt.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary)

#create_model()
load_model()
#train()
predict()

