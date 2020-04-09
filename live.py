import numpy as np 
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

model = load_model('rnn_model.hdf5')

test_data = ImageDataGenerator(rescale = 1./255)

cap = cv2.VideoCapture(1)
count = 0 
results = []
while(True):
    count += 1
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    
##    if count%5 == 0:
    cv2.imwrite('D:\\Projects\\Sign Language Recognition\\jester\\jester\\Only gestures\\predict\\prediction\\asdf' + str(count % 5) + '.jpg', frame)
    if count%10 == 0:
        temp = []
        X_pred = test_data.flow_from_directory('D:\\Projects\\Sign Language Recognition\\jester\\jester\\Only gestures\\predict',shuffle=False, target_size=(100, 176), batch_size=64)
        p = model.predict_generator(X_pred)
        temp.append(np.argmax(p[0]))
        temp.append(np.argmax(p[1]))
        temp.append(np.argmax(p[2]))
        temp.append(np.argmax(p[3]))
        temp.append(np.argmax(p[4]))
        result = temp[3]
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

