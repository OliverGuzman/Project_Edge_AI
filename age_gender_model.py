#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import Dense, Dropout
#%%
images = []
ages = []
genders = []

#Create dataframe for dataset of images
for i in os.listdir('/kaggle/input/utkface-new/crop_part1/'):
    split = i.split('_')
    ages.append(int(split[0]))
    genders.append(int(split[1]))
    
    #Resize image for tensor requirements
    image = load_img('/kaggle/input/utkface-new/crop_part1/'+i)
    image = image.resize((224,224))
    image = np.array(image)
    images.append(image)

df = pd.DataFrame()
df['Images'] = images
df['Ages'] = ages
df['Genders'] = genders
#%%
#Reduce skewness of the dataset
list_of_images = []

for i in range(len(df)):
    if df['Ages'].iloc[i] <= 4:
        list_of_images.append(df.iloc[i])
dataframe_of_images = pd.DataFrame(list_of_images)
dataframe_of_images = dataframe_of_images.sample(frac=0.3)

df = df[df['Ages'] > 4]

df = pd.concat([df, dataframe_of_images], ignore_index = True)

#Convert df into arrays
y_age = np.array(df["Ages"])
y_gender = np.array(df["Genders"])
x_images = np.array([np.array(val) for val in df["Images"]])
#%%
#Initial training for new layer
#Import pre-trained model
base_model_ag = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224,3))

#Remove the last layer and add layer according to article
base_model_ag.trainable = False
x_ag = base_model_ag.layers[-2].output

#Gender head
dense_gender = Dense(256, activation='relu')(x_ag)
output_gender = Dense(1, activation='sigmoid', name='gender_prediction') (dense_gender)

#Age head
dense_age = Dense(256, activation='relu')(x_ag)
dropout_age = Dropout(0.3) (dense_age)
output_age = Dense(1, activation='relu', name='age_prediction') (dropout_age)

model_ag = Model(inputs=base_model_ag.input, outputs=[output_gender, output_age])

model_ag.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=['binary_crossentropy', 'mae'],metrics=["accuracy", "mae"])
model_ag.fit(x=x_images, y=[y_gender,y_age], epochs=1,validation_split=0.2, batch_size=48)
#%%
#Second training for finetunning
model_ag.trainable = True

model_ag.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=['binary_crossentropy', 'mae'],metrics=['accuracy', "mae"])

history = model_ag.fit(x=x_images, y=[y_gender,y_age], 
                    epochs=30,validation_split=0.2,
                    batch_size=48)

#%%
#30 epochs training
# plot results for gender
acc = history.history['gender_prediction_accuracy']
val_acc = history.history['val_gender_prediction_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph Gender')
plt.legend()
plt.figure()

# plot results for age
acc = history.history['age_prediction_mae']
val_acc = history.history['val_age_prediction_mae']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('MAE Graph age')
plt.legend()
plt.figure()

# plot results for loss
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

#%%
#Ten epochs training

# plot results for gender
acc = history.history['gender_prediction_accuracy']
val_acc = history.history['val_gender_prediction_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph Gender')
plt.legend()
plt.figure()

# plot results for age
acc = history.history['age_prediction_mae']
val_acc = history.history['val_age_prediction_mae']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('MAE Graph age')
plt.legend()
plt.figure()

# plot results for loss
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

#%%
#Save model
model_ag.save("models/age_gender_model.keras")
#%%