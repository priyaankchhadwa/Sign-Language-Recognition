import os
from sklearn import svm
import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import libsvm 

libsvm.set_verbosity_wrap(1)

X_train = []
y_train = []

X_test = []
y_test = []

path = os.path.abspath('.')

train_dir = os.listdir('train')

scaler = MinMaxScaler()

c = 0
for i in train_dir:
    for j in os.listdir(os.path.join(path, 'train', i)):
        c += 1
        if c % 1000 == 0:
            print('{}th batch loaded'.format(c / 1000))
        image = cv2.imread(os.path.join(path, 'train', i, j))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = scaler.fit_transform(gray_image)
        resized_image = cv2.resize(gray_image, (150, 100)) 
        flat_image = resized_image.ravel()
        X_train.append(flat_image)
        y_train.append(i)
print('\nTraining data imported...\n')

val_dir = os.listdir('validation')

c = 0
for i in val_dir:
    for j in os.listdir(os.path.join(path, 'validation', i)):
        c += 1
        if c % 1000 == 0:
            print('cth batch loaded')
        image = cv2.imread(os.path.join(path, 'validation', i, j))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = scaler.fit_transform(gray_image)
        resized_image = cv2.resize(gray_image, (150, 100)) 
        flat_image = resized_image.ravel()
        X_test.append(flat_image)
        y_test.append(i)
print('\nValidation data imported...\n')        

labelencoder_y_train = LabelEncoder()
y_train = labelencoder_y_train.fit_transform(y_train)
labelencoder_y_test = LabelEncoder()
y_test = labelencoder_y_test.fit_transform(y_test)
print('labelencoding done...')

new_train_x = list(X_train[:50]) 
new_train_x.extend(X_test[2500:2550])
new_train_x.extend(X_test[-50:])
new_train_x = np.array(new_train_x)

new_train_y = list(y_test[:50]) 
new_train_y.extend(y_test[2500:2550])
new_train_y.extend(y_test[-50:])
new_train_y = np.array(new_train_y)

print('\nstarting svm...')
clf = svm.SVC(verbose=True)
clf.fit(new_train_x, new_train_y)

from sklearn.metrics import accuracy_score
y_true = clf.predict(X_test)
acc = accuracy_score(y_true, y_test)
