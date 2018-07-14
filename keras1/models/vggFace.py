from keras.applications.vgg16 import VGG16 as KerasVGG16
from keras.models import Model
from keras.layers import Flatten, Dense,Input, Dropout
from keras_vggface.vggface import VGGFace

import config
from .base_model import BaseModel


class VGGFACE(BaseModel):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):
        super(VGGFACE, self).__init__(*args, **kwargs)

    def _create(self):

        #custom parameters
        nb_class = len(config.classes)
        hidden_dim = 512

        #base_model = KerasVGG16(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        #self.make_net_layers_non_trainable(base_model)
        
        vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        x = Dense(hidden_dim, activation='relu', name='fc7')(x)
        out = Dense(nb_class, activation='softmax', name='fc8')(x)
        self.model = Model(vgg_model.input, out)

        



def inst_class(*args, **kwargs):
    return VGGFACE(*args, **kwargs)
