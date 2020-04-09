import os
import numpy as np
from keras.preprocessing import image
import pickle

X_train = []
y_train = []

for classes in os.listdir('train'):
    for images in os.listdir(os.path.join('train', classes)):
        tmp = image.load_img(os.path.join('train', classes, images), target_size = (100, 176))
        tmp = image.img_to_array(tmp)
        X_train.append(tmp)
        y_train.append(classes)

X_test = []
y_test = []

for classes in os.listdir('validation'):
    for images in os.listdir(os.path.join('validation', classes)):
        tmp = image.load_img(os.path.join('validation', classes, images), target_size = (100, 176))
        tmp = image.img_to_array(tmp)
        X_test.append(tmp)
        y_test.append(classes)

print('X_train: ', X_train[:5], '\t', X_train[3000:3005], '\t',X_train[-5:])
print('y_train: ', y_train[:5], '\t', y_train[3000:3005], '\t',y_train[-5:])
print('X_test: ', X_test[:5], '\t', X_test[3000:3005], '\t',X_test[-5:])
print('y_test: ', y_test[:5], '\t', y_test[3000:3005], '\t',y_test[-5:])

print(len(X_train), '\t', len(X_test))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

with open('X_train.pkl', 'wb') as file:
    pickle.dump(X_train, file)

with open('y_train.pkl', 'wb') as file:
    pickle.dump(y_train, file)

with open('X_test.pkl', 'wb') as file:
    pickle.dump(X_test, file)

with open('y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)