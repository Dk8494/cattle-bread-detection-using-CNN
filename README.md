
🐄 Cattle Breed Detection Using Deep Learning
📌 Overview

This project implements a Cattle Breed Detection System using Deep Learning and Computer Vision. A trained Convolutional Neural Network (CNN) model classifies cattle images into predefined breeds. The system is deployed using a Streamlit web application for real-time, user-friendly predictions.

🐮 Supported Cattle Breeds

The model detects the following cattle breeds:

Ayrshire Cattle

Jersey Cattle

Holstein Friesian Cattle

Brown Swiss Cattle

Red Dane Cattle

✨ Features

Image-based cattle breed detection

CNN-based deep learning model

Real-time predictions via web app

Displays predicted breed with confidence score

Simple and intuitive user interface

🛠 Technology Stack

Language: Python

Deep Learning: TensorFlow, Keras

Model: Convolutional Neural Network (CNN)

Web Framework: Streamlit

Libraries: NumPy, Matplotlib, PIL

Model Format: HDF5 (.h5)

📂 Project Structure
├── SIH.py                 # Model training script
├── App.py                 # Streamlit application
├── cattle_model1.h5       # Trained CNN model
├── README.md              # Documentation

⚙️ How It Works

Cattle images are collected and organized by breed

Images are resized and normalized

A CNN model is trained to extract visual features

The trained model is saved in .h5 format

The Streamlit app loads the model and predicts the breed from an uploaded image

▶️ How to Run
Install Dependencies
pip install tensorflow streamlit numpy pillow matplotlib

Run the Application
streamlit run App.py

Use the App

Upload a cattle image

View the predicted breed and confidence score

📊 Output

Predicted cattle breed

Confidence percentage

Uploaded image preview

🌾 Applications

Livestock and dairy farm management

Breed identification and verification

Agricultural research

Smart farming solutions

🔮 Future Enhancements

Add more cattle breeds

Mobile app integration

Disease detection in cattle

Cloud deployment

IoT-based farm monitoring
