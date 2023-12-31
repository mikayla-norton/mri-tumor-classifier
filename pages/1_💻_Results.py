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

from warnings import filterwarnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix
import streamlit as st

from PIL import Image
import random

st.set_page_config(page_title='Results - NeuroNet MRI', 
                    page_icon='💻',
                    layout="wide")
st.title("Model Generation Results - NeuroNet MRI")

st.header("Data Preview")
col1, col2, col3, col4 = st.columns([1,1,1,1])

gen = st.button("Generate Random Images of Each Type")
# st.write(gen)

if gen:
    glioma = os.path.join("mri-data/Training/glioma_tumor", random.choice(os.listdir("mri-data/Training/glioma_tumor")))
    meningioma = os.path.join("mri-data/Training/meningioma_tumor", random.choice(os.listdir("mri-data/Training/meningioma_tumor")))
    pituitary = os.path.join("mri-data/Training/pituitary_tumor", random.choice(os.listdir("mri-data/Training/pituitary_tumor")))
    noTumor = os.path.join("mri-data/Training/no_tumor", random.choice(os.listdir("mri-data/Training/no_tumor")))


    image = Image.open(glioma)
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    col1.pyplot(figure)
    col1.subheader("Glioma Tumor")


    image = Image.open(meningioma)
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    col2.pyplot(figure)
    col2.subheader("Meningioma Tumor")

    image = Image.open(pituitary)
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    col3.pyplot(figure)
    col3.subheader("Pituitary Tumor")


    image = Image.open(noTumor)
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    col4.pyplot(figure)
    col4.subheader("No Tumor")

with st.status("Developing Results"):
    st.write("Searching for model")
    # CNN_model = tf.keras.models.load_model('MRI_CNN.keras')
    CNN_h5 = tf.keras.models.load_model('assets/MRI_CNN.h5')
    st.write("Selecting results from model training and predictions")
    result = pd.read_csv("assets/classification_report.csv")  
    image = Image.open("assets/AccuracyLoss.png")
    summary = pd.read_csv("assets/summary.csv")
    st.write("Producing output")

st.write("The figure below displays the changes in training and validation loss and accuracy with each epoch.")
st.image(image)
st.write("The model is trained on the test set and returns the following.")
st.table(result)
st.write("When compared with the expected labels, the confusion matrix holds the following results, with the true values representing the rows and the predictions mapped to the columns.")
col1,col2,col3 = st.columns([1,3,1])
col2.image("assets/CM.png")

with st.expander("For the detailed summary of the CNN model, the information is displayed below."):
    CNN_h5.summary(print_fn=lambda x: st.text(x))