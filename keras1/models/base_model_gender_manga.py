from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from sklearn.externals import joblib

import config
import util
from keras.models import Model
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

class BaseModel(object):
    def __init__(self,
                 class_weight=None,
                 nb_epoch=1000,
                 batch_size = 32,
                 freeze_layers_number=None):
        self.model = None
        self.class_weight = class_weight
        self.nb_epoch = nb_epoch
        self.fine_tuning_patience = 20
        self.batch_size = batch_size
        self.freeze_layers_number = freeze_layers_number
        self.img_size = (224, 224)
        self.loaded_model = False

    def _create(self):
        raise NotImplementedError('subclasses must override _create()')

    def _fine_tuning(self):
        self.freeze_top_layers()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])
        self.model.summary()

        train_data = self.get_train_datagen()
        #train_data = self.get_train_datagen(zca_whitening=True)
        callbacks = self.get_callbacks(config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience)

        if util.is_keras2():
            self.model.fit_generator(
                train_data,
                steps_per_epoch=config.nb_train_samples / float(self.batch_size),
                #steps_per_epoch=1,
                epochs=self.nb_epoch,
                validation_data=self.get_validation_datagen(),
                validation_steps=config.nb_validation_samples / float(self.batch_size),
                callbacks=callbacks,
                class_weight=self.class_weight)
        else:
            self.model.fit_generator(
                train_data,
                samples_per_epoch=config.nb_train_samples,
                #samples_per_epoch=1,
                nb_epoch=self.nb_epoch,
                validation_data=self.get_validation_datagen(),
                nb_val_samples=config.nb_validation_samples,
                callbacks=callbacks,
                class_weight=self.class_weight)

        self.model.save(config.get_model_path())

    #Minha mod
    def test(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])
        self.model.summary()

        train_data = self.get_train_datagen()
        #print(train_data.next())
        #train_data = self.get_train_datagen(rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        callbacks = self.get_callbacks(config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience)

        score = self.model.evaluate_generator(train_data,steps=config.nb_train_samples / float(self.batch_size))
        print("Score: ",score)


    def train(self):
        print("Creating model...")
        if(not self.loaded_model):
            self._create()
            print("Model is created")
        print("Fine tuning...")
        self._fine_tuning()
        self.save_classes()
        print("Classes are saved")

    def extract(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])
        self.model.summary()
        #a = 2/0

        extract_train = False

        if(extract_train):
            train_data = self.get_train_datagen()
        else:
            train_data = self.get_validation_datagen()
        layer_name = 'fc1'
        intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        if(extract_train):
            steps = int(config.nb_train_samples // float(self.batch_size))
        else:
            steps = int(config.nb_validation_samples // float(self.batch_size))
        #print(steps)

        for i in range(steps):
            batch = train_data.next()
            #print(batch)
            #print(batch[0].shape)

            #print(str(batch[1][0]))
            features = intermediate_layer_model.predict_on_batch(batch[0])
            #print(features.shape)
            #a = 2/0
            if(extract_train):
                path = 'train/'
            else:
                path = 'test/'
            #path = 'valid/'

            label =''
            for j in range(len(features)):
                if(str(batch[1][j]) == '[ 1.  0.  0.  0.  0.]'):
                    label = path+'1'
                elif(str(batch[1][j]) == '[ 0.  1.  0.  0.  0.]'):
                    label = path+'2'
                elif(str(batch[1][j]) == '[ 0.  0.  1.  0.  0.]'):
                    label = path+'3'
                elif(str(batch[1][j]) == '[ 0.  0.  0.  1.  0.]'):
                    label = path+'R'
                elif(str(batch[1][j]) == '[ 0.  0.  0.  0.  1.]'):
                    label = path+'W'
                np.save('Features/'+label+'/feature_'+str(i)+'_'+str(j)+'.npy',features[j])
            #a = 2/0

    def evaluate(self):
        print("Evaluate")
        self.model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-5),
        metrics=['accuracy'])
        self.model.summary()
        #a = 2/0

        train_data = self.get_validation_datagen()
        steps = int(config.nb_validation_samples // float(self.batch_size))
        y_true = []
        y_pred = []
        for i in range(steps):
            batch = train_data.next()

            #print(len(batch))
            for j in range(len(batch[1])):
                if(str(batch[1][j]) == '[ 0.  1.]'):
                    y_true.append(1)
                elif(str(batch[1][j]) == '[ 1.  0.]'):
                    y_true.append(0)

            preds= self.model.predict(batch[0])
            for j in range(len(preds)):
                y_prob = preds[j]
                #print(y_prob)
                y_class = np.argmax(y_prob)
                y_pred.append(y_class)

        print(len(y_pred))
        print(len(y_true))
        #print(y_pred)
        #print(y_true)
        #a = 2/0
        print("classification report: ")
        target_names = ['class female', 'class male']
        print(classification_report(y_true, y_pred, target_names=target_names))

        print('Precisions: ')
        print(precision_score(y_true, y_pred, average='macro') )
        print(precision_score(y_true, y_pred, average='micro') )
        print(precision_score(y_true, y_pred, average='weighted'))




    def save(self,s):
        self.model.save(s)

    def load(self):
        print("Creating model")
        self.load_classes()
        self._create()
        self.model.load_weights(config.get_fine_tuned_weights_path())
        self.loaded_model = True
        return self.model

    @staticmethod
    def save_classes():
        joblib.dump(config.classes, config.get_classes_path())

    def get_input_tensor(self):
        if util.get_keras_backend_name() == 'theano':
            return Input(shape=(3,) + self.img_size)
        else:
            return Input(shape=self.img_size + (3,))

    @staticmethod
    def make_net_layers_non_trainable(model):
        for layer in model.layers:
            layer.trainable = False

    def freeze_top_layers(self):
        #print(len(self.model.layers))
        #a = 2/0
        if self.freeze_layers_number:
            print("Freezing {} layers".format(self.freeze_layers_number))
            for layer in self.model.layers[:self.freeze_layers_number]:
                layer.trainable = False
            for layer in self.model.layers[self.freeze_layers_number:]:
                layer.trainable = True

    @staticmethod
    def get_callbacks(weights_path, patience=30, monitor='val_loss'):
        early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
        model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
        return [early_stopping, model_checkpoint]

    @staticmethod
    def apply_mean(image_data_generator):
        """Subtracts the dataset mean"""
        image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

    @staticmethod
    def load_classes():
        config.classes = joblib.load(config.get_classes_path())

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)[0]

    def get_train_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)
        #return idg.flow_from_directory(config.train_dir, target_size=self.img_size, classes=config.classes)
        return idg.flow_from_directory(config.train_dir, target_size=self.img_size, classes=config.classes,batch_size = self.batch_size)

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def get_validation_datagen(self, *args, **kwargs):
        if(config.model == 'inception_v3'):
            print("Pre-processing for incepetion_v3")
            idg = ImageDataGenerator(preprocessing_function= self.preprocess_input)
        else:
            idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)            
        return idg.flow_from_directory(config.validation_dir, target_size=self.img_size, classes=config.classes,batch_size = self.batch_size)
