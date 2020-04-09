from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
from keras.preprocessing import image
import cv2
import numpy as np

base_cnn = load_model('weights-improvement-03-0.340.hdf5')
cnn = Model(inputs=base_cnn.input, outputs=base_cnn.get_layer('global_average_pooling2d_1').output)

rnn = load_model('rnn_model.hdf5')

data = ImageDataGenerator(rescale = 1./255)

def get_prediction():
    X_data = []
    X_pred = data.flow_from_directory('predict', shuffle=False, target_size=(100, 176), batch_size=32)
    features = cnn.predict_generator(X_pred)
    X_data.append(features)
    X_data = np.array(X_data)
    X_data = sequence.pad_sequences(X_data, maxlen=40)   
    result = rnn.predict(X_data)
    
    for i in result:
        return np.argmax(result)

no_of_cap_frames = 30
cap = cv2.VideoCapture(1)
count = 0 
while(True):
    count += 1
    ret, frame = cap.read()
    cv2.imshow('frame1', frame)
   
    cv2.imwrite('predict\\prediction\\' + str(count % no_of_cap_frames) + '.jpg', frame)
    if count%no_of_cap_frames == 0:
        count = 0
        result = get_prediction()
        
        if result == 0:
            print('\nxxxxxxxxxxxxxxxxxxxxx\n Drumming\nxxxxxxxxxxxxxxxxxxxxx\n')
        elif result == 1:
            print('\nxxxxxxxxxxxxxxxxxxxxx \n No gesture\nxxxxxxxxxxxxxxxxxxxxx\n')
        else:
            print('\nxxxxxxxxxxxxxxxxxxxxx\n Swipe up\nxxxxxxxxxxxxxxxxxxxxx\n')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

