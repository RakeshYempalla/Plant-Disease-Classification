from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import pickle
import numpy as np
from flask import jsonify
import json

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'AlexNetModel.hdf5';

global model
# Load your trained model
model = load_model(MODEL_PATH)
#print(model)

# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path):
global model
new_img = image.load_img(img_path, target_size=(224, 224))

# Preprocessing the image
img = image.img_to_array(new_img)
# x = np.true_divide(x, 255)
img = np.expand_dims(img, axis=0)

img = img/224
# Be careful how your trained model deals with the input
# otherwise, it won&#39;t make correct prediction!

prediction = model.predict(img)

d = prediction.flatten()
j = d.max()
li = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
'Apple___healthy', 'Blueberry___healthy&#39;,
'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy', 'Grape___Black_rot',
'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy', 'Potato___Early_blight',
'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
'Tomato___Late_blight', 'Tomato___Leaf_Mold&',
'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus', &'Tomato___healthy']
for index,item in enumerate(d):
if item == j:
class_name = li[index]
print(&quot;Following is our prediction:&quot;,class_name)
return class_name

@app.route('/', methods=['GET','POST'])
def analyze():
if request.method == &quot;POST&quot;:
# Get the file from post request
f = request.files['file']

# Save the file to ./uploads
basepath = os.path.dirname(__file__)
file_path = os.path.join(

31
basepath, 'uploads', secure_filename(f.filename))
f.save(file_path)

# Make prediction
preds = model_predict(file_path)

return render_template('index.html',message=preds)
return render_template('index.html')

if __name__ == '__main__':
app.run(debug=True)
