from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint
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

max_num_data = 3000
temp_file_name = "/tmp/catsvsdogs.hdf5"
batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 32, 32
# number of convolutional filters to use at each layer
nb_filters = [32, 64]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]
# the CIFAR10 images are RGB
image_dimensions = 3

def load_data():
    image_path = "/Users/quinnjarrell/datasets/catsvsdogs/train/resized/"
    onlyfiles = [ f for f in listdir(image_path) if isfile(join(image_path,f)) and f != '.DS_Store']
    
    nb_train_samples = min(int(len(onlyfiles) * 0.8), max_num_data)
    nb_test_samples = min(int(len(onlyfiles) - (len(onlyfiles) * 0.8)), max_num_data)

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    X_test = np.zeros((nb_test_samples, 3, 32, 32), dtype="uint8")
    y_test = np.zeros((nb_test_samples,), dtype="uint8")

    i = 0
    for file_name in onlyfiles:
        full_name = join(image_path, file_name)
        if i == nb_train_samples:
            break
        category = 1 if file_name[:3] == 'cat' else 0
        pic = Image.open(full_name)
        y_train[i] = category
        X_train[i] = img_to_array(pic)
         
        pic.close()
        i += 1
    i = 0
    for file_name in onlyfiles[nb_train_samples:]:
        full_name = join(image_path, file_name)
        if i == nb_test_samples:
            break
        category = 1 if file_name[:3] == 'cat' else 0
        pic = Image.open(full_name)
        y_test[i] = category
        X_test[i] = img_to_array(pic)
         
        pic.close()
        i += 1

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))


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
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters[0], nb_filters[0], nb_conv[0], nb_conv[0]))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[0], nb_pool[0])))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters[1], nb_filters[0], nb_conv[0], nb_conv[0], border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters[1], nb_filters[1], nb_conv[1], nb_conv[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[1], nb_pool[1])))
model.add(Dropout(0.25))

model.add(Flatten())
# the image dimensions are the original dimensions divided by any pooling
# each pixel has a number of filters, determined by the last Convolution2D layer
model.add(Dense(nb_filters[-1] * (shapex / nb_pool[0] / nb_pool[1]) * (shapey / nb_pool[0] / nb_pool[1]), 512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


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
        store_to_s3(str(int(time.time())), f.read())
        

def load_model():
    items = get_bucket_items()
    newest_item = max(map(int,items))
    model_weights = get_from_s3(str(newest_item))
    with open('./current.weights', 'w') as f:
        f.write(model_weights)
    model.load_weights("./current.weights")

def check_for_early_shutdown(x):
    if x % 7 == 0:
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
        store_to_s3(str(int(time.time())),f.read())

def train():
    
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    #checkpointer = ModelCheckpoint(filepath="/Users/quinnjarrell/Desktop/Experiments/keras/saved/", verbose=1, save_best_only=True)
    min_score = 91293921
    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(X_train)
        progbar = generic_utils.Progbar(X_train.shape[0])
        x = 0
        for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=128):# save_to_dir="/Users/quinnjarrell/datasets/catsvsdogs/train/resized/resized_generated"):
            loss = model.train_on_batch(X_batch, Y_batch)
            x += 1
            check_for_early_shutdown(x)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, Y_test, batch_size=128):
            score = model.test_on_batch(X_batch, Y_batch)
            x += 1
            check_for_early_shutdown(x)
            progbar.add(X_batch.shape[0], values=[("test loss", score)])
        if score < min_score:
            print ("New best model with score: %s", score)
            save_data()
            min_score = score

load_model()
train()
#create_model()