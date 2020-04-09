from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers 
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

num_class = 3
col = 176
row = 100

from keras.callbacks import ModelCheckpoint
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

callbacks_list = [checkpoint, early_stopping]


train_data = ImageDataGenerator(rescale = 1./255,
                                samplewise_std_normalization=True, 
                                shear_range = 0.2, 
                                zoom_range = 0.1)

test_data = ImageDataGenerator(rescale = 1./255)

X_train = train_data.flow_from_directory('train',
                                         target_size = (row, col),
                                         batch_size = 128,
                                         class_mode = 'categorical') 
label_map = (X_train.class_indices)
print(label_map)

X_test = test_data.flow_from_directory('validation',
                                       target_size = (row, col),
                                       batch_size = 128,
                                       class_mode = 'categorical') 
label_map1 = (X_test.class_indices)
print(label_map1)
   
input_shape = (row, col, 3)

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, 
          activation='relu',
          kernel_regularizer=regularizers.l2(0.01), 
          name = 'FC_sftm')(x)
x = Dropout(0.25)(x)


predictions = Dense(num_class, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
#model.load_model('tp2.model')
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999), 
              loss='categorical_crossentropy', 
              metrics = ['accuracy'])

# train the model on the new data for a few epochs
hist1 = model.fit_generator(X_train,
                            steps_per_epoch = 9000/64,
                            epochs = 4,
                            validation_data = X_test,
                            validation_steps = 1773)            

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:

#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
#from keras.optimizers import ADAM
model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999), 
              loss='categorical_crossentropy', 
              metrics = ['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
hist2 = model.fit_generator(X_train,
                       steps_per_epoch = (9000/64),
                       epochs = 10,
                       validation_data = X_test,
                       validation_steps = 1773, 
                       initial_epoch = 2, 
                       callbacks = callbacks_list)



model.save('incep.h5')

#test_data = ImageDataGenerator(rescale = 1./255)
#X_pred = test_data.flow_from_directory('C:/Users/nachiket/Desktop/SEM_8/BE_project/Codes/predict',shuffle = False,target_size = (size, size),batch_size = 64,class_mode = None)       #Creates a DirectoryIterator object  for getting images from the directory specified with images in sub directories 0,1,2,3,4 for train set 
#p = model.predict_generator(X_pred)     #predicting for single image
