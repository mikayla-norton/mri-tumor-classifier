import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

from warnings import filterwarnings
import matplotlib.pyplot as plt
import seaborn as sns

############ SET GRAPHING PARAMS ###############
plt.rcParams.update({'text.color': "white",
                    'axes.labelcolor': "white",
                    'axes.edgecolor': 'white',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'figure.facecolor': '0F1116',
                    'axes.facecolor': '0F1116'})


################# PAGE LAYOUT ##################

st.set_page_config(layout="wide")
st.title("About - Tumor Classification from MRI Scans")

col1, col2, col3= st.columns([2,2,2])

############### PROJECT WRITE UP ###############

col2.image("mri-img.png")

col1.header("Project Background")

col1.write("Magnetic Resonance Imaging (MRI) is a common and crucial diagnostic tool in medicine to scan and image the human anatomy. In neuroscience, MRIs are used regularly to review potential for brain disorders, aneurysms, tumors, and other neurologic conditions. Machine Learning poses new potential as a tool to be involved in interpretation of these images, including location and type of tumor, resulting in potential changes in treatment plans.")
col1.write("In this study, we explore four possible brain tumor diagnostics: glioma tumors (developing from glial cells), pituitary tumors (growth on the pituitary glands), meningioma tumors (growth on the membranes enclosing the brain and spinal cord), and lastly, patients with no tumor. The course of treatment for these tumors varies widely, as gliomas often progress quickly and aggressively, however pituitary tumors and meningiomas are often benign or very slow to progress.")
col1.write("Using a convolutional neural network (CNN), this project aims to predict the type of tumor from an MRI image. A successful machine learning classifier could lead to quicker diagnostics and provide a better prognosis for patients.")

col1.header("Methods")

col1.write("This project applies a convolutional neural network to a training and testing set of MRI images to generate a classification model for MRI scan diagnostics.")

col1.subheader("Data Pre-Processing")
col1.markdown("#### Image Loading")
col1.write("Images from different classes (glioma_tumor, no_tumor, meningioma_tumor, pituitary_tumor) are loaded from the training and testing directories using OpenCV (cv2). The images are resized to a common size (150x150 pixels) using cv2.resize().")

col2.markdown("#### Train-Test Split and One-Hot Encoding")
col2.write("The data is split into training and testing sets using train_test_split from scikit-learn. Labels are encoded into one-hot vectors using a mapping dictionary.")

col2.subheader("Model Architecture")
col2.markdown("#### Base Model - EfficientNetB0")
col2.write("EfficientNetB0 is used as the base model for feature extraction, and is pre-trained on ImageNet.")

col2.markdown("#### Custom Model Head")
col2.write("A custom head is added on top of the base model. It consists of a Global Average Pooling layer, a Dropout layer, and a Dense layer with softmax activation.")

col2.markdown("#### Model Compilation")
col2.write("The model is compiled using categorical crossentropy loss and the Adam optimizer to improve accuracy.")

col3.subheader("Model Training")

col3.markdown("#### Callbacks")
col3.write("During training, three callbacks are used - TensorBoard, ModelCheckpoint, and ReduceLROnPlateau.")

col3.markdown("#### Training")
col3.write("The model is trained using the fit method with training data. Validation data is taken from a subset of the training data.")

col3.subheader("Model Analysis")
col3.markdown("#### Visualization")
col3.write("The training and validation accuracy/loss are visualized using matplotlib and seaborn.")

col3.markdown("#### Prediction and Evaluation")
col3.write("The model is used to predict on the test data, and the predictions are converted back from one-hot encoding. The results are displayed and can be used for further evaluation.")


col3.header("Acknowledgements")
col3.write("Data source: Kaggle Brain Tumor Classification (MRI)")
col3.write("Gratitude to Dr. Murillo and Teaching Assistant Mahyar Abedi for the mentorship throughout the CMSE 830 semester and fostering encouragement through the duration of this project.")



############### PROJECT OUTPUTS ###############
