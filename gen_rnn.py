import numpy as np
import os
import shutil
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
import pickle

#base_model = load_model('weights-improvement-03-0.61.hdf5')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten_1').output)

base_model = load_model('inception-improvement-new-06-0.79.hdf5')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d_1').output)

data = ImageDataGenerator(rescale = 1./255)

X_data = []
y_data = []

def remove_zeros(p):
    tmp = []
    for i in p:
        j = i[i>0]
        tmp.append(j)
    tmp = np.array(tmp)
    return tmp

    
def get_prediction(label,vid_id):
    X_pred = data.flow_from_directory('predict',
                                       shuffle=False,
                                       target_size=(100, 176),
                                       batch_size=64)

    p = model.predict_generator(X_pred)
    #np.save('rnn_pickles/'+label+'_'+vid_id,p)
    #p = remove_zeros(p)
    #p = sequence.pad_sequences(p, maxlen=40)
    X_data.append(p)
    y_data.append(label)
    
test_path = 'rnn_data\\validate\\'
train_path = 'rnn_data\\train\\'
pred_path = 'predict\\'

path = test_path

for classes in os.listdir(path):
    for videos in os.listdir(os.path.join(path,classes)):
        shutil.move(os.path.join(path,classes,videos),os.path.join(pred_path,videos))
        get_prediction(classes,videos)
        shutil.move(os.path.join(pred_path,videos),os.path.join(path,classes,videos))

X_data = np.array(X_data)
X_data = sequence.pad_sequences(X_data, maxlen=40)
y_data = np.array(y_data)
data = [X_data, y_data]

with open("rnn_pickles\\rnn_test20.pickle","wb") as f:
	pickle.dump(X_data,f)      


path = train_path

for classes in os.listdir(path):
    for videos in os.listdir(os.path.join(path,classes)):
        shutil.move(os.path.join(path,classes,videos),os.path.join(pred_path,videos))
        get_prediction(classes,videos)
        shutil.move(os.path.join(pred_path,videos),os.path.join(path,classes,videos))

X_data = np.array(X_data)
X_data = sequence.pad_sequences(X_data, maxlen=40)
y_data = np.array(y_data)
data = [X_data, y_data]

with open("rnn_pickles\\rnn_train200.pickle","wb") as f:
    pickle.dump(X_data,f)      

