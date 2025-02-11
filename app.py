import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import pandas as pd
from pathlib import Path
import os.path

train_data_path = Path("/home/blockchain/Downloads/FIsh classification/images.cv_jzk6llhf18tm3k0kyttxz/data/train")

filepaths = list(train_data_path.glob(r'**/*.jpg'))
labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1] , filepaths))

filepaths = pd.Series(filepaths,name='Filepaths').astype(str)
labels = pd.Series(labels,name='Lables')

train_df =  pd.concat([filepaths,labels],axis=1)

class_names = sorted(train_df['Lables'].unique()) 

def load_image(img):
    image_load = image.load_img(img,target_size=(224,224))
    img_arr = image.img_to_array(image_load)
    img_arr = np.expand_dims(img_arr,axis = 0)
    img_arr = img_arr/255.0
    return img_arr

model_path = "/home/blockchain/Downloads/FIsh classification/fish_classification_model.keras"

model = tf.keras.models.load_model(model_path)


st.title("🐟 Fish Classification App")
st.write("Upload a fish image, and the model will predict the category.")

upload_file = st.file_uploader("upload an image to predict",type = ['jpg','png','jpeg'])

if upload_file is not None:
    img_arr = load_image(upload_file)
    prediction = model.predict(img_arr)
    predicted_class_idx = np.argmax(prediction,axis = 1)[0]
    predicted_class = class_names[predicted_class_idx]
    confidence_score = prediction[0]

    st.image(upload_file,caption = "uploaded image",use_container_width=True)
    st.subheader(f"🔍 Predicted Class: **{predicted_class}**")

    st.write("🔢 **Confidence Scores:**")
    for label, score in zip(class_names, confidence_score):
        st.write(f"- {label}: **{score:.4f}**")

    
    st.success(f"🎯 Model is **{confidence_score[predicted_class_idx] * 100:.2f}%** confident in this prediction.")