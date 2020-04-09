from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import os

X_train = []

model = load_model('incep.h5')  # create the original model

#model.summary()
layer_name = 'global_average_pooling2d_2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)


v3_train_path_drum = 'D:\\Projects\\Sign Language Recognition\\jester\\jester\\20bn-jester-v3\\train\\Drumming Fingers'

drum = dict()

images = os.listdir(v3_train_path_drum)
for i in images:
	key = i.split('_')[0]
	drum[key] = []
for j in images:
	key = j.split('_')[0]
	if len(drum[key]) < 30:
		drum[key].append(j)
			
v3_train_path_no = 'D:\\Projects\\Sign Language Recognition\\jester\\jester\\20bn-jester-v3\\train\\No Gesture'

no = dict()

images = os.listdir(v3_train_path_no)
for i in images:
	key = i.split('_')[0]
	no[key] = []
for j in images:
	key = j.split('_')[0]
	if len(no[key]) < 30:
		no[key].append(j)
		
v3_train_path_swipe = 'D:\\Projects\\Sign Language Recognition\\jester\\jester\\20bn-jester-v3\\train\\Swiping Up'

swipe = dict()

images = os.listdir(v3_train_path_swipe)
for i in images:
	key = i.split('_')[0]
	swipe[key] = []
for j in images:
	key = j.split('_')[0]
	if len(swipe[key]) < 30:
		swipe[key].append(j)
		
for key, value in drum.items():
	sequence = []
	for i in value:
		img = image.load_img(os.path.join(v3_train_path_drum, i), target_size=(100,176))
		data = image.img_to_array(img)
		data = np.expand_dims(data, axis=0)
		data = intermediate_layer_model.predict(data)
		sequence.append(data)
	X_train.append(sequence)







img = image.load_img(os.path.join(v3_train_path, classes, i), target_size=(100,176))
data = image.img_to_array(img)
data = np.expand_dims(data, axis=0)
data = intermediate_layer_model.predict(data)
sequence.append(data)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		