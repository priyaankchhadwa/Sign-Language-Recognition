from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
import pickle
import numpy as np
from keras.utils import to_categorical

with open('rnn_pickles\\rnn_train200.pickle','rb')as f:
    X_train = pickle.load(f)

with open('rnn_pickles\\rnn_test20.pickle','rb')as f:
    X_test = pickle.load(f)
    
y_train = np.zeros(597)
y_train[198:397] = 1
y_train[397:597] = 2
# y_train[600:800] = 3

y_test = np.zeros(120)
y_test[40:80] = 1
y_test[80:120] = 2
# y_test[120:160] = 3

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(LSTM(2048, return_sequences=False,
               input_shape=(40,2048),
               dropout=0.2))
#model.add(LSTM(2048, return_sequences=True))
#model.add(LSTM(2048))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])

model.fit(X_train, y_train,
			batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
            epochs=15)

model.save('rnn_model.hdf5')
