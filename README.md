# NeuroNet MRI - MRI Brain Tumor Classifier

A convolutional neural network built in Keras/TensorFlow to classify MRI scans into tumor vs. non-tumor, with an interactive Streamlit front end.

---

## Overview

This project trains a CNN on labeled MRI slices to detect brain tumors.  
It includes:  
- **Data:** Brain MRI images in `mri-data/`  
- **Model:** Keras-based CNN defined in `home.py`  
- **Demo:** Streamlit app launched from `home.py`  
- **Logs:** Training artifacts stored in `logs/`  

---


## Live Demo

Access the interactive Streamlit application to explore the model’s performance and upload your own:

[NeuroNet MRI](https://mri-tumor-classifier.streamlit.app/)

---

## Architecture

1. **Data Loading & Preprocessing**  
   - Read and normalize MRI slices from `mri-data/`  
   - Split into train/test folds  

2. **Model Definition** (`home.py`)  
   - Convolutional layers with batch normalization and dropout  
   - Fully connected classifier head  

3. **Training & Evaluation**  
   - Loss: Categorical cross-entropy  
   - Metrics: Accuracy, precision, recall  
   - Checkpoints and logs written to `logs/`  

4. **Streamlit Interface**  
   - User uploads an MRI image  
   - Model predicts “Tumor” vs. “No Tumor”  
   - Display of prediction and confidence score  

---

## Project Structure
```
mri-tumor-classifier/
├── assets/ # Icons and static images for the app
├── logs/ # Model checkpoints & training logs
├── mri-data/ # MRI scan dataset (train/test splits)
├── pages/ # Subpages of Streamlit app
├── home.py # Main Streamlit script & model pipeline
├── requirements.txt# Python dependencies
└── .gitignore # Patterns for files to ignore
```


---

## Technologies

- **Frameworks:** TensorFlow, Keras, Streamlit  
- **Libraries:** NumPy, Pandas, Matplotlib, scikit-learn  
- **Languages:** Python 3.10+  

---

## Contact

For questions or feedback, please contact Mikayla Norton at [mikayla.e.norton@gmail.com](mailto:mikayla.e.norton@gmail.com).

