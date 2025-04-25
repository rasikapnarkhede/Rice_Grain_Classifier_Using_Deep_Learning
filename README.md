# Rice_Grain_Classifier_Using_Deep_Learning

Rice Image Classifier
This project is an image classification system that identifies different types of rice grains using Convolutional Neural Networks (CNN). A simple user interface (UI) is built using Streamlit so users can upload rice images and see predictions.

Objective
Build a deep learning model to classify different rice grain types using a pre-trained CNN and provide predictions via a user-friendly web app.

Dataset
Rice Image Dataset - https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset
Contains images of different rice types in separate folders
Each folder name is the rice type label

Environment Setup
pip install tensorflow streamlit numpy pandas opencv-python matplotlib scikit-learn pillow

How to Run the App
streamlit run app.py

Upload Image
Use the upload button in the web app
It will show:
The uploaded image
Predicted rice grain type

Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
