from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.models import Model

import numpy as np
import os


model = load_model('trained/model-vgg16.h5')
model.summary()
model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)

sources = ['data/sorted/train/1','data/sorted/train/2','data/sorted/train/3','data/sorted/train/W','data/sorted/train/R']
destines = ['data/features/train/1','data/features/train/2','data/features/train/3','data/features/train/W','data/features/train/R']

for i in range(len(sources)):
	s = sources[i]
	d = destines[i]
	#if(s != 'ImbalanceClasses/2'):
	#	continue
	files = os.listdir(s)
	for f in files:
		print("In folder: "+s +" and in file: "+f)
		#if(os.path.isfile(d+'/'+f+'.npy')):
		#	continue
		img = image.load_img(s+'/'+f, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		#x = preprocess_input(x)

		preds = model.predict(x)
		print(np.argmax(preds))
		#features = model_extractfeatures.predict(x)
		#print(features.shape)
		#a = 2/0
		#np.save(d+'/'+f+'.npy',features)
	a = 2/0