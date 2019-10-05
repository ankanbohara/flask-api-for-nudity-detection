from flask import Flask, render_template,request,redirect,url_for
from werkzeug import secure_filename
import tensorflow as tf
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image as immage
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
import cv2
from sklearn.model_selection import train_test_split
import os
import sys
import pydot
from sklearn.externals import joblib
from matplotlib.pyplot import imshow 
import keras.backend as K
from keras.models import model_from_json
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')
	
#Function to upload files
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
   	f = request.files['image']
   	fname = secure_filename(f.filename)
   	f.save(fname)
   	return redirect(url_for('predictor',filename=fname))

#Function to give predictions
@app.route('/predict/<filename>')
def predictor(filename):
	K.clear_session() 
	json_file = open('model/model3.json', 'r')
	loaded_model_json = json_file.read()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model/model3.h5")
	print("Loaded model from disk")
		# imaage = send_from_directory(filename)
	img = immage.load_img(filename, target_size=(224, 224))
	x = immage.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	print('Input image shape:', x.shape)
	print(loaded_model.predict(x))
	a = loaded_model.predict(x)[0];
	ans = a[1]*100.0
	return "Model predicted the image to be of nudity : %.4f precentage"%ans

if __name__ == '__main__':
   app.run(debug = True)

