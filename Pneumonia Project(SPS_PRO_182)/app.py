#PNEUMONIA PREDICTION USING X-RAY by Ayush Patel

#importing libraries
from __future__ import division, print_function
import os
import glob
import numpy as np
from keras.preprocessing import image 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras import backend
from tensorflow.keras import backend
import tensorflow as tf
#from skimage.transform import resize
# Flask utilities
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

global graph
graph=tf.get_default_graph()


app = Flask(__name__)

# importing x-ray prediction model
MODEL_PATH = 'models/model.h5'

# Loading trained model...
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


#starting routing...

@app.route('/', methods=['GET'])
def index():
    # Home page showcasing...
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
    
        f = request.files['file']

        # Saving the file to ./uploads directory
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        #loading the image and resizing...
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            predictions = model.predict_classes(x)    
        index = ['Pneumonia not detected.','Pneumonia detected!']
        i = predictions.flatten()
        text = index[i[0]]
        
        return text
    


if __name__ == '__main__':
    app.run(debug=False,threaded = False)


