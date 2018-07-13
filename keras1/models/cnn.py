from keras.applications.vgg16 import VGG16 as KerasVGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import config
from .base_model import BaseModel


class CNN(BaseModel):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)

    def _create(self):
        """
        base_model = KerasVGG16(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='elu', name='fc1')(x)
        x = Dropout(0.6)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.6)(x)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)

        self.model = Model(input=base_model.input, output=predictions)
        """

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=(224,224)+(3,)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(config.classes), activation='softmax', name='predictions'))
        self.model = model
        


def inst_class(*args, **kwargs):
    return CNN(*args, **kwargs)
