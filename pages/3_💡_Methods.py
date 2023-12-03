import streamlit as st


################# PAGE LAYOUT ##################

st.set_page_config(page_title='Methods - NeuroNet MRI', 
                    page_icon='💡',
                    layout="wide")
st.title("Methods - NeuroNet MRI")

col1, col2, col3= st.columns([2,2,2])

############### PROJECT WRITE UP ###############

col2.image("assets/mri-img.png")

col1.header("Methods")

col1.write("This project applies a convolutional neural network to a training and testing set of MRI images to generate a classification model for MRI scan diagnostics.")

col1.subheader("Data Pre-Processing")
col1.markdown("#### Image Loading")
col1.write("Images from different classes (glioma_tumor, no_tumor, meningioma_tumor, pituitary_tumor) are loaded from the training and testing directories using OpenCV (cv2). The images are resized to a common size (150x150 pixels) using cv2.resize().")

col1.markdown("#### Train-Test Split and One-Hot Encoding")
col1.write("The data is split into training and testing sets using train_test_split from scikit-learn. Labels are encoded into one-hot vectors using a mapping dictionary.")

col1.subheader("Model Architecture")
col1.markdown("#### Base Model - EfficientNetB0")
col1.write("EfficientNetB0 is used as the base model for feature extraction, and is pre-trained on ImageNet.")

col2.markdown("#### Custom Model Head")
col2.write("A custom head is added on top of the base model. It consists of a Global Average Pooling layer, a Dropout layer, and a Dense layer with softmax activation.")

col2.markdown("#### Model Compilation")
col2.write("The model is compiled using categorical crossentropy loss and the Adam optimizer to improve accuracy.")

col2.subheader("Model Training")

col2.markdown("#### Callbacks")
col2.write("During training, three callbacks are used - TensorBoard, ModelCheckpoint, and ReduceLROnPlateau.")

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

