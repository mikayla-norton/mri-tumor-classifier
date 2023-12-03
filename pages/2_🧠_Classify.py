import numpy as np
import cv2

import tensorflow as tf



import streamlit as st

from PIL import Image


st.set_page_config(page_title='Classify - NeuroNet MRI', 
                    page_icon='🧠',
                    layout="wide")
st.title("Classify - NeuroNet MRI")

file = st.file_uploader("Please upload an MRI scan: ", type=["jpg", "png", "jpeg"])
url = "https://drive.google.com/drive/folders/1hBKM8E6SPLeYVhU7wuAIWDhhfd5EiEAn?usp=sharing"
st.write("~ Don't have a image to upload? Try one of [these](%s) ~" % url)
st.text ("[files displayed come from an un-tested and un-trained set of data to be used for exploration]")

if file is not None:

    st.subheader("Uploaded Image Preview:")
    image = Image.open(file)
    st.image(image, width=500)
    with st.spinner(text="Generating Results... please wait..."):

        image = image.save("assets/img.jpg")
        image = cv2.imread("assets/img.jpg")
        image_size = 150
        image = cv2.resize(image,(image_size,image_size))
        image_batch = np.expand_dims(image, axis=0)


        CNN_model = tf.keras.models.load_model('assets/MRI_CNN.h5')
        prediction = CNN_model.predict(image_batch)
        p = CNN_model.predict_generator(image_batch)
        st.write(p)
        y_class = np.argmax(prediction,axis=1)


        dict = {0 : 'glioma tumor', 1 : 'no tumor', 2: 'meningioma tumor', 3 : 'pituitary tumor'}

        y_class = list(map(dict.get, y_class, y_class))[0]


        st.markdown("#### Predicted Class: "+ y_class)
        st.write("The untrained eye cannot quickly identify the presence or classification of neuro tumors, however, the CNN developed in NeuroNet MRI has the ability to detect and classify new images with high accuracy.")

