from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, img_to_array,array_to_img
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
from worker import store_to_s3, get_from_s3, get_bucket_items, shutdown_spot_request
import tempfile
import time
import StringIO
import requests
import theano
import os
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
debug_mode = os.environ.get('LOCALDEBUG')
api_key = os.environ['SENDGRID']
sendgrid_url = "https://api.sendgrid.com/api/mail.send.json"
sendgrid_data = """api_user=quinnjarr&api_key=your_sendgrid_password&to=destination@example.com&toname=Destination&subject=Example_Subject&text=testingtextbody&from=info@domain.com"""
startup_data={"to" : "quinnjarr@gmail.com",
              "from" : "quinnjarr@gmail.com",
              "subject" : "Startup",
              "html" : "Starting ec2 cats vs dogs",
              }
authent_header = {"Authorization" : "Bearer %s" % api_key}
if debug_mode is None:
    requests.post(sendgrid_url, data=startup_data, headers=authent_header)
max_num_data = 300000
temp_file_name = "/tmp/catsvsdogs.hdf5"
batch_size = 32
nb_classes = 2
nb_epoch = 10
data_augmentation = True

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 32, 32
# number of convolutional filters to use at each layer
nb_filters = [32, 32]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = (2, 2)
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = (3, 3)
# the CIFAR10 images are RGB
image_dimensions = 3

def load_data():
    image_path = os.environ['DATA_PATH']
    onlyfiles = [ f for f in listdir(image_path) if isfile(join(image_path,f)) and f != '.DS_Store']
    
    nb_train_samples = 9000#min(int(len(onlyfiles) * 0.8), max_num_data)
    nb_test_samples = 500#min(int(len(onlyfiles) - (len(onlyfiles) * 0.8)), max_num_data)

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    X_test = np.zeros((nb_test_samples, 3, 32, 32), dtype="uint8")
    y_test = np.zeros((nb_test_samples,), dtype="uint8")
    cat_imgs = []#np.zeros((nb_test_samples, 3, 32, 32), dtype="uint8")
    dog_imgs = []#np.zeros((nb_test_samples, 3, 32, 32), dtype="uint8")
    i = 0
    for file_name in onlyfiles:
        
        category = 1 if file_name[:3] == 'cat' else 0
        if category == 1:
            cat_imgs.append(file_name) #X_train[i]
        else:
            dog_imgs.append(file_name) #X_train[i]

    for i in range(nb_train_samples):
        file_name = cat_imgs[i] if i % 2 == 0 else dog_imgs[i]
        full_name = join(image_path, file_name)
        pic = Image.open(full_name)
        y_train[i] = i % 2 == 0
        X_train[i] = img_to_array(pic)
        pic.close()

    cat_slice = cat_imgs[nb_train_samples:]
    dog_slice = dog_imgs[nb_train_samples:]
    for i in range(nb_test_samples):
        file_name = cat_slice[i] if i % 2 == 0 else dog_slice[i]
        full_name = join(image_path, file_name)
        pic = Image.open(full_name)
        y_test[i] = i % 2 == 0
        #print( i % 2 == 0)        
        X_test[i] = img_to_array(pic)
        pic.close()

    """for file_name in onlyfiles[nb_train_samples:]:
        full_name = join(image_path, file_name)
        if i == nb_test_samples:
            break
        category = 1 if file_name[:3] == 'cat' else 0
        pic = Image.open(full_name)
        y_test[i] = category
        X_test[i] = img_to_array(pic)
         
        pic.close()
        i += 1"""


    y_train = np.reshape(y_train, (len(y_train), 1))

    y_test = np.reshape(y_test, (len(y_test), 1))

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    return (X_train, y_train), (X_test, y_test)

# the data, shuffled and split between tran and test sets
print("Loading training data")
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

print("compiling")
mode = 'FAST_COMPILE' if debug_mode else 'FAST_RUN'
print("Using %s mode" % mode)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', theano_mode=mode)
bucket_name = "catsvsdogs"
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.best_lost = 9239129
        self.i = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch == nb_epoch:
            shutdown_spot_request()
        loss = logs.get('val_loss')
        print("%s loss is %s" % (self.i, loss))
       
        if loss < self.best_lost:
            self.i += 1
            print(self.i)
            self.best_lost = loss
            if self.i % 3 == 0:
                save_data()
                save_email_data = startup_data
                save_email_data['html'] = "new loss is %s" % loss
                save_email_data['subject'] = "New saved!"
                if debug_mode is None:
                    requests.post(sendgrid_url, data=save_email_data, headers=authent_header)

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        check_for_early_shutdown()
        
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
        store_to_s3(str(int(time.time())),bucket_name, f.read())
        

def load_model():
    print("Getting bucket items")
    items = get_bucket_items(bucket_name)
    newest_item = max(map(int,items))
    print("Newest item %s, retrieving now" % newest_item)
    model_weights = get_from_s3(str(newest_item), bucket_name)
    print("loading model")
    with open('./current.weights', 'w') as f:
        f.write(model_weights)

    model.load_weights("./current.weights")


shutdown_data = startup_data
shutdown_data['html'] = "Shutting down cats vs dogs"
shutdown_data['subject'] = "Early shutdown" 
def check_for_early_shutdown():
    try:
        resp = requests.get("http://169.254.169.254/latest/meta-data/spot/termination-time", timeout=0.1)
        if resp.status == 200:
            save_data() 
            if debug_mode is None:
                requests.post(sendgrid_url, data=shutdown_data, headers=authent_header)
            return 
    except:
        pass
    

def save_data():
    model.save_weights(temp_file_name, overwrite=True)
    with open(temp_file_name, 'r') as f:
        store_to_s3(str(int(time.time())), bucket_name, f.read())

def train():
    
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    #checkpointer = ModelCheckpoint(filepath="/Users/quinnjarrell/Desktop/Experiments/keras/saved/", verbose=1, save_best_only=True)
    history = LossHistory()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=100000, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test),callbacks=[history])
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    if debug_mode is None:
        shutdown_spot_request()
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
    print("It is of category ", model.predict(X_test[50:55]))
    print(Y_test[50:55])
    import pdb; pdb.set_trace()  # breakpoint 8d9fb711 //

    Y_pred = model.predict(X_test)
    # Convert one-hot to index
    y_pred = np.argmax(Y_pred, axis=1)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    # Visualize convolution result (after activation)
    i = 100
    # Visualize the first layer of convolutions on an input image
    X = X_test[i:i+1]
    print("I predict a ", model.predict(X))
    
    plt.figure()
    plt.title('input')

    nice_imshow(plt.gca(), array_to_img(np.squeeze(X)), vmin=0, vmax=1, cmap=cm.binary)
    C1 = convout1_f(X)
    C1 = np.squeeze(C1)
    print("C1 shape : ", C1.shape)

    plt.figure(figsize=(shapex, shapey))
    plt.suptitle('convout1')
    #nice_imshow(plt.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary)

create_model()
#load_model()
#train()
#predict()

