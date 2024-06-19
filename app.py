from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense,Input
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

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

# Load the pre-trained model
# model = keras.models.load_model('model.h5')

model = create_model()
model.load_weights('model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def predict(image_path):
    img = Image.open(image_path)
    input_img = np.asarray(img.resize((160,160)))
    result = model.predict(np.asarray([input_img]))

    situation=["sitting","using_laptop","hugging","sleeping","drinking",
           "clapping","dancing","cycling","calling","laughing"
          ,"eating","fighting","listening_to_music","running","texting"]

    # Make prediction
    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]
    probability = np.max(result)*100
    action = situation[prediction]
    return probability, action

@app.route('/submit', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        img = request.files['my_image']
        # Process the uploaded image
        img_path = img.stream
        image_path = os.path.join('static', img.filename)
        img.save(image_path)
        probability, action = predict(img_path)
        

    return render_template('index.html', prediction = action, probability = probability, img_path = image_path)

@app.route("/", methods = ['GET', 'POST'])
def home():
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)