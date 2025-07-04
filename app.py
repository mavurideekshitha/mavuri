from flask import Flask, render_template,  request,url_for
import pandas as pd
import pickle
from typing import Tuple
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
import sqlite3
import shutil




model3 = pickle.load(open("0-3.sav", "rb"))
model11 = pickle.load(open("4-11.sav", "rb"))
MODEL_PATH = 'image_model.tflite'
modelmri=load_model('monument_classifier.h5')
with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)

modeladhd=load_model('adhd.h5')
with open('adhd.pkl', 'rb') as f:
            class_adhd = pickle.load(f)

app = Flask(__name__)

def get_interpreter(model_path: str) -> Tuple:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def predict(image_path: str) -> int:
    interpreter, input_details, output_details = get_interpreter(MODEL_PATH)
    input_shape = input_details[0]['shape']
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, (input_shape[2], input_shape[2]))
    img = tf.expand_dims(img, axis=0)
    resized_img = tf.cast(img, dtype=tf.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], resized_img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    return np.argmax(results, axis=0)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Three_year', methods=['GET', 'POST'])
def Three_year():
    if request.method == 'POST':
        df = request.form
        data = []
        data.append(int(df['A1']))
        data.append(int(df['A2']))
        data.append(int(df['A3']))
        data.append(int(df['A4']))
        data.append(int(df['A5']))
        data.append(int(df['A6']))
        data.append(int(df['A7']))
        data.append(int(df['A8']))
        data.append(int(df['A9']))
        data.append(int(df['A10']))

        if int(df['age']) < 12:
            data.append(0)
        else:
            data.append(1)
        
        data.append(int(df['gender']))

        if df['etnicity'] == 'middle eastern':
            data.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'White European':	
            data.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'Hispanic':
            data.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'black':
            data.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'asian':	
            data.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'south asian':
            data.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Native Indian':
            data.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if df['etnicity'] == 'Others':	
            data.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if df['etnicity'] == 'Latino':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])	
        if df['etnicity'] == 'mixed':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if df['etnicity'] == 'Pacifica':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        data.append(int(df['Jaundice']))
        data.append(int(df['ASD']))

        name = df['name']
        email = df['email']
        print(name)
        print(email)

        Index = model3.predict([data])
        if Index == 0:
            prediction = 'Non-autistic'
        else:
            prediction = 'Autistic'
        print(prediction)

        return render_template('index.html', name=name, email=email,  prediction=prediction)
    return render_template('index.html')

@app.route('/Eleven_year', methods=['GET', 'POST'])
def Eleven_year():
    if request.method == 'POST':
        df = request.form
        data = []
        data.append(int(df['A1']))
        data.append(int(df['A2']))
        data.append(int(df['A3']))
        data.append(int(df['A4']))
        data.append(int(df['A5']))
        data.append(int(df['A6']))
        data.append(int(df['A7']))
        data.append(int(df['A8']))
        data.append(int(df['A9']))
        data.append(int(df['A10']))

        if int(df['age']) < 12:
            data.append(0)
        else:
            data.append(1)
        
        data.append(int(df['gender']))

        if df['etnicity'] == 'Others':
            data.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Middle Eastern':	
            data.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'Hispanic':
            data.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'White-European':
            data.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Black':	
            data.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'South Asian':
            data.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])	
        if df['etnicity'] == 'Asian':
            data.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if df['etnicity'] == 'Pasifika':	
            data.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if df['etnicity'] == 'Turkish':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        if df['etnicity'] == 'Latino':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        data.append(int(df['Jaundice']))
        data.append(int(df['ASD']))

        name = df['name']
        email = df['email']
        print(name)
        print(email)

        Index = model11.predict([data])
        if Index == 0:
            prediction = 'Non-autistic'
        else:
            prediction = 'Autistic'
        print(prediction)

        return render_template('index.html', name=name, email=email,  prediction=prediction)
    return render_template('index.html')

@app.route('/Image', methods=['GET', 'POST'])
def Image():
    if request.method == 'POST':
        name = request.form['name']
        filename = request.form['filename']
        email = request.form['email']
        path = 'static/test/'+filename
        Index = predict(path)

        print(name)
        if Index == 1:
            prediction = 'Non-autistic'
        else:
            prediction = 'Autistic'
        print(prediction)

        return render_template('index.html', name=name, email=email, prediction=prediction, img='http://127.0.0.1:5000/'+path)
    return render_template('index.html')

@app.route('/mri_image', methods=['GET', 'POST'])
def mri_image():
    if request.method == 'POST':
        name = request.form['name']
        filename = request.form['filename']
        email = request.form['email']
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        

        shutil.copy("static/testmri/"+fileName, dst)
        image = cv2.imread("satic/testmri/"+fileName)

        # model=load_model('monument_classifier.h5')
        path='static/images/'+fileName


        # # Load the class names
        # with open('class_names.pkl', 'rb') as f:
        #     class_names = pickle.load(f)
        dec=""
        dec1=""
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = modelmri.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        print(predicted_class, confidence)
        if predicted_class == 'AUTISTIC':
            str_label = "AUTISTIC"

           
        elif predicted_class == 'NORMAL':
            str_label = "NORMAL"

            
        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"

        return render_template('index.html', name=name, email=email, prediction=str_label, img='http://127.0.0.1:5000/'+path)
    return render_template('index.html')


@app.route('/adhd_image', methods=['GET', 'POST'])
def adhd_image():
    if request.method == 'POST':
        name = request.form['name']
        filename = request.form['filename']
        email = request.form['email']
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        

        shutil.copy("static/testhand/"+fileName, dst)
        image = cv2.imread("satic/testhand/"+fileName)

        # model=load_model('monument_classifier.h5')
        path='static/images/'+fileName


        # # Load the class names
        # with open('class_names.pkl', 'rb') as f:
        #     class_names = pickle.load(f)
        dec=""
        dec1=""
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = modeladhd.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            print(f"\n\n\n {class_adhd} \n\n\n")
            predicted_class = class_adhd[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        print(predicted_class, confidence)
        if predicted_class == 'healthy':
            str_label = "NO ADHD"

           
        elif predicted_class == 'spiral_adhd':
            str_label = "ADHD IN SPIRAL"
        elif predicted_class == 'wave_adhd':
            str_label = "ADHD IN WAVE"

            
        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"

        return render_template('index.html', name=name, email=email, prediction=str_label, img='http://127.0.0.1:5000/'+path)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)