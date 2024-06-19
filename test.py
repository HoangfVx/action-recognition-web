import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense,Input
from PIL import Image
import numpy as np

def create_model():

    model = Sequential()

    pretrained_model= tf.keras.applications.VGG16(include_top=False,
                    input_shape=(160,160,3),
                    pooling='avg',classes=15,
                    weights='imagenet')


    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(15, activation='softmax'))
    return model


model = create_model()
model.load_weights('model.h5')

image = Image.open('./Human Action Recognition/test/Image_10.jpg')
input_img = np.asarray(image.resize((160,160)))
result = model.predict(np.asarray([input_img]))

situation=["sitting","using_laptop","hugging","sleeping","drinking",
           "clapping","dancing","cycling","calling","laughing"
          ,"eating","fighting","listening_to_music","running","texting"]
itemindex = np.where(result==np.max(result))
print('itemindex:{}'.format(itemindex))
prediction = itemindex[1][0]
print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", situation[prediction])