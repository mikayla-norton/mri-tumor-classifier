import streamlit as st
from PIL import Image

import matplotlib.pyplot as plt

############ SET GRAPHING PARAMS ###############
plt.rcParams.update({'text.color': "white",
                    'axes.labelcolor': "white",
                    'axes.edgecolor': 'white',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'figure.facecolor': '0F1116',
                    'axes.facecolor': '0F1116'})


################# PAGE LAYOUT ##################

st.set_page_config(page_title='NeuroNet MRI', 
                                    page_icon=':house:', 
                                    layout='wide')

st.title("NeuroNet MRI")

st.header("Unlock Clarity, Enhance Care: Revolutionizing Tumor Detection with Precision MRI Analysis.")

c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11= st.columns(11)
c1.image(Image.open('assets/image-logos/python.png'))
c2.image(Image.open('assets/image-logos/keras.png'))
c3.image(Image.open('assets/image-logos/tf.png'))
c4.image(Image.open('assets/image-logos/opencv.png'))
c5.image(Image.open('assets/image-logos/streamlit.png'))
c6.image(Image.open('assets/image-logos/sklearn.png'))
c7.image(Image.open('assets/image-logos/tqdm.png'))
c8.image(Image.open('assets/image-logos/pandas.png'))
c9.image(Image.open('assets/image-logos/seaborn.png'))
c10.image(Image.open('assets/image-logos/numpy.png'))
c11.image(Image.open('assets/image-logos/matplotlib.png'))

c1, c2 = st.columns(2)

left_co1, cent_co1,last_co1 = c1.columns([1,6,1])
left_co2, cent_co2,last_co2 = c2.columns([1,6,1])


c1.subheader("Revolutionizing Tumor Detection")
c1.write("The integration of Convolutional Neural Networks (CNN) in MRI image classification heralds a significant shift in medical diagnostics. This technology enhances the ability to accurately detect and categorize brain tumors, surpassing traditional methods in both speed and precision. The use of CNN in analyzing complex MRI data means more reliable diagnoses, reducing uncertainty and potentially expediting treatment decisions. This advancement not only improves diagnostic accuracy but also paves the way for personalized treatment plans, making it a game-changer in the field of neuroimaging and patient care.")

with cent_co2:
    st.image("assets/home-imgs/img1.png")

c1, c2 = st.columns(2)

left_co1, cent_co1,last_co1 = c1.columns([1,6,1])
left_co2, cent_co2,last_co2 = c2.columns([1,6,1])
c2.subheader("Advanced AI, Enhanced Diagnosis")
c2.write("The CNN classifier in NeuroNet MRI significantly enhances diagnostic precision in detecting brain tumors. By utilizing deep learning algorithms, it meticulously analyzes MRI images, identifying subtle patterns and anomalies that may be overlooked in traditional methods. This precision allows for earlier and more accurate detection of a variety of tumor types, leading to timely and targeted treatment interventions. This advanced diagnostic tool, therefore, represents a leap forward in neuroimaging, offering healthcare professionals a powerful ally in the fight against brain tumors.")

with cent_co1:
    st.image("assets/home-imgs/img2.png")

c1, c2 = st.columns(2)

left_co1, cent_co1,last_co1 = c1.columns([1,6,1])
left_co2, cent_co2,last_co2 = c2.columns([1,6,1])

c1.subheader("Comprehensive Tumor Classification")
c1.write("Utilizing advanced CNN algorithms, NeuroNet AI not only detects the presence of a tumor with 99.8% accuracy from saggital, axial, and coronal MRI scans, but also classifies it into specific types: meningioma, glioma, or pituitary tumor. This precise categorization is crucial for developing targeted treatment strategies, as each tumor type requires a different approach. By providing this detailed analysis, our app empowers medical professionals with critical insights for patient diagnosis and treatment planning, making it an invaluable tool in modern neurology and oncology.")

with cent_co2:
    st.image("assets/home-imgs/img3.png")

c1, c2 = st.columns(2)

left_co1, cent_co1,last_co1 = c1.columns([1,6,1])
left_co2, cent_co2,last_co2 = c2.columns([1,6,1])
c2.subheader("Speed and Efficiency in Diagnosis")
c2.write("NeuroNet AI significantly accelerates the diagnostic process for brain tumors. By harnessing the power of advanced CNN technology, it rapidly analyzes MRI scans, delivering results in a fraction of the time required by traditional methods. This speed ensures that healthcare professionals can make quicker, more informed decisions about patient care, reducing the waiting period for diagnoses and facilitating earlier commencement of treatment. In the fast-paced medical environment, our app is an essential tool, streamlining the diagnostic workflow and enhancing patient management efficiency.")
with cent_co1:
    st.image("assets/home-imgs/img4.png")

c1, c2 = st.columns(2)

left_co1, cent_co1,last_co1 = c1.columns([1,6,1])
left_co2, cent_co2,last_co2 = c2.columns([1,6,1])

c1.subheader("Getting Started with NeuroNet MRI")
c1.write("Want to learn more about the model?")
c1.write("Explore the comprehensive results of model training on the results tab.")
c1.write("Ready to get started?")
c1.write("Head over to the classifier tab to upload an MRI scan for classification now.")
c1.write("Want to learn how NeuroNet MRI was made?")
c1.write("Check out the methods tab for more.")

cent_co2.write("")
with cent_co2:
    st.image("assets/home-imgs/img5.png")

