import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
from tensorflow.keras.preprocessing import image
import os
import pandas as pd 
import random
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import itertools
import h5py
import io
import pickle
from keras.models import load_model
from keras.models import Model
# Deep learning libraries
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
a=st.sidebar.radio("Pick a Disease",["Home","pnemonia","Malaria","Heart_disease"])
if a=="Home":
    st.title("Disease predictor")
if a=="Malaria":
    st.write('malaria')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    @st.cache(allow_output_mutation=True,suppress_st_warning=True)
    def load_cnn1():
        model_ = load_model('weights1.h5')
        return model_

    @st.cache(allow_output_mutation=True,suppress_st_warning=True)
    def load_cnn2():
        model_ = load_model('weights3.h5')
        return model_

    def preprocessed_image(file):
        image = file.resize((44,44), Image.ANTIALIAS)
        image = np.array(image)
        image = np.expand_dims(image, axis=0) 
        return image

    def display_activation(activations, col_size, row_size, act_index): 
        activation = activations[act_index]
        activation_index=0
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
        for row in range(0,row_size):
            for col in range(0,col_size):
                ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
        st.pyplot(fig)


    def main():
        st.title('CNN for Classification Malaria Cells')
        st.sidebar.title('Web Apps using Streamlit')
        menu = {1:"Prediction"}
        def format_func(option):
            return menu[option]
        choice= st.sidebar.selectbox("Menu",options=list(menu.keys()), format_func=format_func)
    
        if choice == 1:
            st.subheader("CNN Models")
            st.markdown("#### Complex CNN and Simple CNN")
        
            models = st.sidebar.radio(" Select model to perform prediction", ("Complex CNN", "Simple CNN"))
            if models=="Complex CNN":
                model_1 = load_cnn1()

            
                st.subheader('Test on an Image')
                images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
                if images is not None:
                    images = Image.open(images)
                    st.text("Image Uploaded!")
                    st.image(images,width=300)
                    used_images = preprocessed_image(images)
                    predictions = np.argmax(model_1.predict(used_images), axis=-1)
                    if predictions == 1:
                        st.error("Cells get parasitized")
                    elif predictions == 0:
                        st.success("Cells is healty Uninfected")
                
                    st.sidebar.subheader('Visualization in Complex CNN')
                    layer_outputs = [layer.output for layer in model_1.layers]
                    activation_model = Model(inputs=model_1.input, outputs=layer_outputs)
                    activations = activation_model.predict(used_images.reshape((1,44,44,3)))
                    layers = st.sidebar.slider('Which layer do you want to see ?', 0, 18, 0, format="no %d ")
                    st.subheader('Visualize Layer')
                    if layers == 1 :
                        st.write("Layers Conv_1")
                        display_activation(activations, 8, 4, 0)
                    elif layers == 2 :
                        st.write("Layers Activation_1 (ReLU)")
                        display_activation(activations, 8, 4, 1)
                    elif layers == 3 :
                        st.write("Layers Conv_2")
                        display_activation(activations, 8, 4, 2)
                    elif layers == 4 :
                        st.write("Layers Activation_2 (ReLU)")
                        display_activation(activations, 8, 4, 3)
                    elif layers == 5 :
                        st.write("Layers Max_pooling2d_1")
                        display_activation(activations, 8, 4, 4)
                    elif layers == 6 :
                        st.write("Layers Batch_normalization_1")
                        display_activation(activations, 8, 4, 5)
                    elif layers == 7 :
                        st.write("Layers Dropout_1")
                        display_activation(activations, 8, 4, 6)
                    elif layers == 8 :
                        st.write("Layers Conv_3")
                        display_activation(activations, 8, 8, 7)
                    elif layers == 9 :
                        st.write("Layers Activation_3 (ReLU)")
                        display_activation(activations, 8, 8, 8)
                    elif layers == 10 :
                        st.write("Layers Conv_4")
                        display_activation(activations, 8, 8, 9)
                    elif layers == 11 :
                        st.write("Layers Max_pooling2d_2")
                        display_activation(activations, 8, 8, 10)
                    elif layers == 12 :
                        st.write("Layers Batch_normalization_2")
                        display_activation(activations, 8, 8, 11)
                    elif layers == 13 :
                        st.write("Layers Dropout_2")
                        display_activation(activations, 8, 8, 12)
                    elif layers == 14 :
                        st.write("Layers Conv_5")
                        display_activation(activations, 16, 8, 13)
                    elif layers == 15 :
                        st.write("Layers Activation_4 (ReLU)")
                        display_activation(activations, 16, 8, 14)
                    elif layers == 16 :
                        st.write("Layers Conv_6")
                        display_activation(activations, 16, 16, 15)
                    elif layers == 17 :
                        st.write("Layers Batch_normalization_3")
                        display_activation(activations, 16, 16, 16)
                    elif layers == 18 :
                        st.write("Layers Dropout_3")
                        display_activation(activations, 16, 16, 17)
        
            elif models=="Simple CNN":
                model_2 = load_cnn2()
                st.subheader('Test on an Image')
                images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
                if images is not None:
                    images = Image.open(images)
                    st.text("Image Uploaded!")
                    st.image(images,width=300)
                    used_images = preprocessed_image(images)
                    predictions = np.argmax(model_2.predict(used_images), axis=-1)
                    if predictions == 1:
                        st.error("Cells get parasitized")
                    elif predictions == 0:
                        st.success("Cells is healty Uninfected")
                
                    st.sidebar.subheader('Visualization in Simple CNN')
                    layer_outputs = [layer.output for layer in model_2.layers]
                    activation_model = Model(inputs=model_2.input, outputs=layer_outputs)
                    activations = activation_model.predict(used_images.reshape((1,44,44,3)))
                    layers = st.sidebar.slider('Which layer do you want to see ?', 0, 6, 0, format="no %d ")
                    st.subheader('Visualize Layer')
                    if layers == 1: 
                        st.write("Layers Conv_1")
                        display_activation(activations, 8, 4, 0)
                    elif layers == 2 :
                        st.write("Layers Max_pooling2d_1")
                        display_activation(activations, 8, 4, 1)
                    elif layers == 3 :
                        st.write("Layers Conv_2")
                        display_activation(activations, 8, 8, 2)
                    elif layers == 4 :
                        st.write("Layers Max_pooling2d_2")
                        display_activation(activations, 8, 8, 3)
                    elif layers == 5 :
                        st.write("Layers Batch_normalization_1")
                        display_activation(activations, 8, 8, 4)
                    elif layers == 6 :
                        st.write("Layers Dropout_1")
                        display_activation(activations, 8, 8, 5)
            # Change path of the weights file
    @st.cache(allow_output_mutation=True,suppress_st_warning=True)
    def load_cnn1():
        model_ = load_model('weights1.h5')
        return model_

    @st.cache(allow_output_mutation=True,suppress_st_warning=True)
    def load_cnn2():
        model_ = load_model('weights3.h5')
        return model_
        
    if __name__ == "__main__":
        main()

if a=="pnemonia":
    loaded_model=tf.keras.models.load_model('my_model.h5')
    #st.set_option('depreciation.showfileUploaderEncoding',False)
    #@st.cache(allow_output_mutation)
    st.title("Pneumonia Detection Machine")
    file=st.sidebar.file_uploader("Please upload your X-Ray image and Nothing Else", type= ["png","JPG","jpeg"])
    def predict(image_path):
        image1 = image.load_img(image_path, target_size=(150, 150))
        image1 = image.img_to_array(image1)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        #st.write(image1.shape)
        img_array= image1/255
        prediction = loaded_model.predict(img_array)
        if prediction[0][0]>.6:
            string="You have a high chance of having Pneoumonia, Please consult a doctor"
        else:
            string="You have a low chance of having Pneoumonia, Nothing to panic!"
        st.success(string)
    if file is not None:
        img=Image.open(file).convert('RGB')
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        predict(file)


if a=="Heart_disease":
    st.write('Heart_disease')
    # loading in the model to predict on the data
    pickle_in = open('heart.pkl','rb')
    heart = pickle.load(pickle_in)
    def prediction(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
        prediction = heart.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        print(prediction)
        return prediction
        # this is the main function in which we define our webpage
    def main():
            # giving the webpage a title
        st.title("Heart Disease Decector")
            # here we define some of the front end elements of the web page like
            # the font and background color, the padding and the text to be displayed
        html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Heart Disease Dector </h1>
	</div>"""
            # this line allows us to display the front end aspects we have
	        # defined in the above code
        st.markdown(html_temp, unsafe_allow_html = True)
            # the following lines create text boxes in which the user can enter
            # the data required to make the prediction
        age = st.text_input("age", "Type Here")
        sex = st.text_input("sex", "1 or 0")
        cp= st.text_input("cp", "1 or 0")
        trestbps = st.text_input("trestbps", "50-180")
        chol = st.text_input("chol", "100-300")
        fbs = st.text_input("fbs", "0 or 1")
        restecg = st.text_input("restecf", "0-5")
        thalach = st.text_input("thalach", "100-200")
        exang = st.text_input("exang", "0 or 1")
        oldpeak = st.text_input("oldpeak", "0-7")
        slope = st.text_input("slope", "0-3")
        ca = st.text_input("ca", "0-5")
        thal = st.text_input("thal", "0-6")
            #target= st.text_input("target", "0 or 1")
        result =""
        if st.button("Predict"):
            result = prediction(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
            print(result)
            if result == 1:
                st.success("There is chance of Heart disease")
            else:
                st.success("There is no chance of Heart disease")
    if __name__=='__main__':
        main()
                
    
