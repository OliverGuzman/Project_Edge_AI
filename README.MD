# Project Edge AI

This is the code used in the project for the task 2: Age, gender, and expression recognition application

# What is this?

This project focuses in investigating, training, testing, and using two deep learning models in the edge that determine age, gender and emotion of a person through an app for mobile device.

## Structure

The project is divided in two parts, the app that holds Java code and utilizes the trained models to run predictions, and the python code to preprocess the data and train the models.

The app consists of three main classes MainActivity.java, CameraActivity.java, and qualities_recognition.java. In summary, these three clases connect the interface of the app with the camera and the models to run the predictions using a real time feed and return the prediction and draw a rectangule around the face which includes the predictions.

For the python code, it consists of three files age_gender_model.py, emotion_model.py, and model_convertion.py. In the first two files, it is described the preprocessing of data, model structure, model trianing and plotting of the results for both emotion and age and gender models. The thrid file describes the testing of the models before and after the converstion to tensorflow lite.

## Installation for python code.

After the creation and later activation of a virtual enviroment, please go to the root and install the requirments:

```shell
pip install -r requirements.txt
```

## Installation for app.

This app was developed using Android Studio, Java, and OpenCv 4.9.0. Therefore, I will recommend reviewing a tutorial on how to import a project into android studio and install 4.9.0 given how extensive the process is.
