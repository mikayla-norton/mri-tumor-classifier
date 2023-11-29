import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras import optimizers
import pickle

from warnings import filterwarnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix
import streamlit as st

from PIL import Image
from skimage import transform


st.set_page_config(layout="wide")
st.title("Tumor Classification Upload")

file = st.file_uploader("Please upload an MRI scan: ", type=["jpg", "png", "jpeg"])
url = "https://drive.google.com/drive/folders/1hBKM8E6SPLeYVhU7wuAIWDhhfd5EiEAn?usp=sharing"
st.write("~ Don't have a image to upload? Try one of [these](%s) ~" % url)
st.text ("[files displayed come from an un-tested and un-trained set of data to be used for exploration]")

if file is not None:
    st.subheader("Uploaded Image Preview:")
    image = Image.open(file)
    st.image(image, width=500)
    np_image = np.array(image).astype('float32')
    image_size = 150
    img = cv2.resize(np_image,(image_size,image_size))
    # np_image = np.expand_dims(np_image, axis=0)

    CNN_model = tf.keras.models.load_model('MRI_CNN.h5')
    prediction = CNN_model.predict(img)
    y_class = np.argmax(prediction,axis=1)
    # dict = {0 : 'glioma_tumor', 1 : 'no_tumor', 2: 'meningioma_tumor', 3 : 'pituitary_tumor'}

    # y_class = list(map(dict.get, y_class, y_class))[0]


    st.write("Predicted Class: ", y_class)

# CNN_model = tf.keras.saving.load_model("CNN.h5")
