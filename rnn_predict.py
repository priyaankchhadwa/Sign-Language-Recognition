import numpy as np
import os
import shutil
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence

base_cnn = load_model('incep.h5')
cnn = Model(inputs=base_cnn.input, outputs=base_cnn.get_layer('global_average_pooling2d_2').output)

rnn = load_model('rnn_pickles/rnn_200_80.h5')

data = ImageDataGenerator(rescale = 1./255)

X_data = []

   
def get_prediction():
    X_pred = data.flow_from_directory('predict',
                                       shuffle=False,
                                       target_size=(100, 176),
                                       batch_size=64)

    p = cnn.predict_generator(X_pred)
    X_data.append(p)
    
val_path = 'rnn_pickles/val'
pred_path = 'predict/'

path = val_path

for classes in os.listdir(path):
    for videos in os.listdir(os.path.join(path,classes)):
        shutil.move(os.path.join(path,classes,videos),os.path.join(pred_path,videos))
        get_prediction()
        shutil.move(os.path.join(pred_path,videos),os.path.join(path,classes,videos))

X_data = np.array(X_data)
X_data = sequence.pad_sequences(X_data, maxlen=40)

print(X_data.shape)
result = rnn.predict(X_data)
print(result)
