#Import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Dense
#%%
#load dataset
from datasets import load_dataset

#Transfrom to dataframe, and shuffle
df_emotion = load_dataset("imagefolder", data_dir="/kaggle/input/affectnet-training-data")
df_emotion = pd.DataFrame(df_emotion["train"])
df_emotion = df_emotion.rename(columns={"image": "Images", "label": "Labels"})
df_emotion = df_emotion.sample(frac = 1)

#transform image to array, and resize.
def toArray(x):
    image = x.resize((224,224))
    image = np.array(image)
    return image

df_emotion["Images"] = df_emotion["Images"].map(toArray) 

#Extract label and create class weights for training
(unique, counts) = np.unique(df_emotion["Labels"], return_counts=True)
cw=1/counts
cw/=cw.min()
class_weights = {i:cwi for i,cwi in zip(unique,cw)}

#Divide dataset into train and test
df_emotion_train = df_emotion.iloc[:22540,:]
df_emotion_test = df_emotion.iloc[22540:,:]
#%%
## convert label to categorical
le = LabelEncoder()
le.fit(df_emotion_train['Labels'])
y_train_e = le.transform(df_emotion_train['Labels'])
y_test_e = le.transform(df_emotion_test['Labels'])

y_train_e = to_categorical(y_train_e, num_classes=8)
y_test_e = to_categorical(y_test_e, num_classes=8)

#%%
#Verify array shapes
x_images_emotion_train = np.array([np.array(val) for val in df_emotion_train["Images"]])
x_images_emotion_test = np.array([np.array(val) for val in df_emotion_test["Images"]])

#%%
#Initial training for new layer
#Import pre-trained model
base_model_e = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224,3))

#Remove the last layer and add layer according to article
base_model_e.trainable = False
x_e = base_model_e.layers[-2].output

#Gender head
dense_emotion = Dense(256, activation='relu')(x_e)
output_emotion = Dense(8, activation='softmax', name='emotion_prediction') (dense_emotion)

model_e = Model(inputs=base_model_e.input, outputs=output_emotion)

model_e.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=['categorical_crossentropy'],metrics=['accuracy','mae'])
model_e.fit(x=x_images_emotion_train, 
            y=y_train_e, epochs=1,
            validation_data=[x_images_emotion_test,y_test_e], 
            class_weight=class_weights,
            batch_size=48)

#%%

#Second training for finetunning
model_e.trainable = True

model_e.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=['categorical_crossentropy'],metrics=['accuracy','mae'])

history_e = model_e.fit(x=x_images_emotion_train, 
                        y=y_train_e, epochs=30,
                        validation_data=[x_images_emotion_test,y_test_e], 
                        class_weight=class_weights,
                        batch_size=48,)

#%%

#30 epochs training
# plot results for gender
acc = history_e.history['accuracy']
val_acc = history_e.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph Emotion')
plt.legend()
plt.figure()

acc = history_e.history['mae']
val_acc = history_e.history['val_mae']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('MAE Graph Emotion')
plt.legend()
plt.figure()

# plot results for loss
loss = history_e.history['loss']
val_loss = history_e.history['val_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

#%%

#Ten epochs training
# plot results for emotion
acc = history_e.history['accuracy']
val_acc = history_e.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

# plot results for loss
loss = history_e.history['loss']
val_loss = history_e.history['val_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()
#%%
#Save model
model_e.save("models/emotion_model.keras")
#