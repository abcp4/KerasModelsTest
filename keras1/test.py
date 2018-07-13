# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.applications.imagenet_utils import preprocess_input

from sklearn.metrics import log_loss
import os 
import numpy as np 
from keras.preprocessing import image
from random import shuffle
from keras.utils import to_categorical
import util
import config
import argparse
import traceback
from keras.optimizers import Adam


classes_number = 5 


def init():
    util.lock()
    util.set_img_format()
    util.override_keras_directory_iterator_next()
    util.set_classes_from_train_dir()
    util.set_samples_info()

    if util.get_keras_backend_name() != 'theano':
        util.tf_allow_growth()

    if not os.path.exists(config.trained_dir):
        os.mkdir(config.trained_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to data dir')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture', choices=[
        config.MODEL_RESNET50,
        config.MODEL_RESNET152,
        config.MODEL_INCEPTION_V3,
        config.MODEL_VGG16])
    parser.add_argument('--nb_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--freeze_layers_number', type=int, help='will freeze the first N layers and unfreeze the rest')
    return parser.parse_args()

def load_images(files):
    x_train = []
    for f in files:
    # if data are in form of images
        img = image.load_img(f[0], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # preprocessing if required
        x_train.append(preprocess_input(x)[0])
    # finally converting list into numpy array
    x_train = np.asarray(x_train)
    x_train = np.squeeze(x_train)
    #x_validation = np.asarray(x_validation)

    return x_train

def load_target(files):
    y_train = []
    for f in files:
        y_train.append(f[1])
    y_train = np.asarray(y_train) 
    
    return to_categorical(y_train, num_classes=classes_number)

def imageLoader(files, batch_size):

    L = len(files)

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_images(files[batch_start:limit])
            Y = load_target(files[batch_start:limit])
            if(len(X)!=batch_size):
                print("Break")
                break

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    batch_size = 16 
    epochs = 10
    path = 'data/sorted/'
    source_test = [path+'test/1',path+'test/2',path+'test/3',path+'test/W',path+'test/R']
    files = []
    y_train = []
    for i in range(len(source_test)):
        s = source_test[i]
        fs= os.listdir(s)
        for j in range(len(fs)):
            f = fs[j]
            f = s+'/'+f
            fs[j] = f
            y_train.append(i)
        files+=fs
    test_data = zip(files,y_train)
    #print(list(train_data))
    #print(len(list(train_data)))
    test_data = list(test_data)
    shuffle(test_data)
    print("Test size: ",len(test_data))
    v = (len(test_data)//batch_size)
    print("Loaded images paths and labels!")
    try:
        args = parse_args()
        if args.data_dir:
            config.data_dir = args.data_dir
            config.set_paths()
        if args.model:
            config.model = args.model

        init()

        model = util.get_model_class_instance(
            class_weight=util.get_class_weight(config.train_dir),
            nb_epoch=args.nb_epoch,
            batch_size=args.batch_size,
            freeze_layers_number=args.freeze_layers_number)

        model.load()
        print("Loaded Model! Now compiling...")
        model.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])
        model.model.summary()
        print("Compiled! now evaluating")
        model.model.evaluate_generator(imageLoader(test_data,batch_size),steps = v)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        util.unlock()
    