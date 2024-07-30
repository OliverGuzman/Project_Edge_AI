#%%
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
from fdlite import FaceDetection, FaceDetectionModel

#Load models
detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
age_gender_model= tf.keras.models.load_model('models/age_gender_model.keras')
emotion_model = tf.keras.models.load_model('models/emotion_model.keras')

#%%
#Load image and perform predictions
image = cv2.imread('datasets/Affectnet/anger/image0000060.jpg')
faces = detect_faces(image)

if not len(faces):
    print('no faces detected :(')
else:
    rect = faces[0].bbox.absolute(size=(96,96))
    faces=image[int(rect.ymin):int(rect.ymax),int(rect.xmin):int(rect.xmax)]
    faces=cv2.resize(faces,dsize=(224,224))
    faces=np.reshape(faces,(1,224,224,3))
    
    preds_e = emotion_model.predict(faces)
    preds_ag = age_gender_model.predict(faces)

    print(preds_e, preds_ag)
#%%
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
tflite_model = converter.convert()
#%%
# Save the model.
with open('emotion_model.tflite', 'wb') as f:
  f.write(tflite_model)

# %%
#Load tensorflow lite model version
interpreter_ag = tf.lite.Interpreter(model_path = "models/age_gender_model.tflite")
interpreter_e = tf.lite.Interpreter(model_path = "models/emotion_model.tflite")

#%%

#Verify the model input and output
input_details_ag = interpreter_ag.get_input_details()
output_details_ag = interpreter_ag.get_output_details()
print("Input Shape:", input_details_ag[0]['shape'])
print("Input Type:", input_details_ag[0]['dtype'])
print("Output Shape:", output_details_ag[0]['shape'])
print("Output Type:", output_details_ag[0]['dtype'])
print("Output Shape:", output_details_ag[1]['shape'])
print("Output Type:", output_details_ag[1]['dtype'])

input_details_e = interpreter_e.get_input_details()
output_details_e = interpreter_e.get_output_details()
print("Input Shape:", input_details_e[0]['shape'])
print("Input Type:", input_details_e[0]['dtype'])
print("Output Shape:", output_details_e[0]['shape'])
print("Output Type:", output_details_e[0]['dtype'])
# %%

#Perform predictions using tensorflow lite models
interpreter_ag.allocate_tensors()
interpreter_e.allocate_tensors()

faces = detect_faces(image)

if not len(faces):
    print('no faces detected :(')
else:
    rect = faces[0].bbox.absolute(size=(96,96))
    faces=image[int(rect.ymin):int(rect.ymax),int(rect.xmin):int(rect.xmax)]
    faces=cv2.resize(faces,dsize=(224,224))
    faces=np.reshape(faces,(1,224,224,3))
    faces = np.float32(faces)

    interpreter_ag.set_tensor(input_details_ag[0]['index'],faces)
    interpreter_ag.invoke()
    tflite_model_predictions_ag = interpreter_ag.get_tensor(output_details_ag[0]['index'])
    print("Prediction results age:", tflite_model_predictions_ag)
    tflite_model_predictions_ag = interpreter_ag.get_tensor(output_details_ag[1]['index'])
    print("Prediction results gender:", tflite_model_predictions_ag)

    interpreter_e.set_tensor(input_details_e[0]['index'], faces)
    interpreter_e.invoke()
    tflite_model_predictions_e = interpreter_e.get_tensor(output_details_e[0]['index'])
    print("Prediction results emotion:", tflite_model_predictions_e)
# %%
