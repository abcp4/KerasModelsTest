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

batch_size = 5
batch = np.zeros((batch_size, 224,224,3),dtype='uint8')
for i in range(len(sources)):
	s = sources[i]
	d = destines[i]
	print("In source : ",s)
	#if(s != 'ImbalanceClasses/2'):
	#	continue
	files = os.listdir(s)
	files_len = len(files);
	count = 0
	while(count<files_len):
		print("Begin loop")
		#batch = []
		b_count = 0
		for j in range(batch_size):
			if(count+j>= files_len):
				break
			f = files[count+j]
			print("In folder: "+s +" and in file: "+f)
			#if(os.path.isfile(d+'/'+f+'.npy')):
			#	continue
			img = image.load_img(s+'/'+f, target_size=(224, 224))
			#x = image.img_to_array(img)
			#x = np.expand_dims(x, axis=0)
			#x = preprocess_input(x)
			#print(x.shape)
			#print(batch.shape)
			#batch.append(x)
			batch[j,:,:,:]= img
			b_count+=1
		count+=b_count
		features = model_extractfeatures.predict_on_batch(batch)
		#print(features.shape)
		t = -batch_size
		#print("Saving features")
		for feat in features:
			if(count+t> files_len):
				break

			f = files[count+t]
			np.save(d+'/'+f+'.npy',feat.reshape((4096,1)))
			t+=1
		#a = 2/0