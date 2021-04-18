#importing dependencies
import numpy as np
from matplotlib import pyplot as plt
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator

#LNET MODEL
model= Sequential()
model.add(Conv2D(6,(5,5),activation='tanh',input_shape=(32,32,3)))
model.add(AveragePooling2D())
model.add(Conv2D(16,(5,5),activation='tanh'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


#Visualizing model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()
#Pre compilation before training model.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255)
train=r'D:\FaceMaskDetector-master\train'
test=r'D:\FaceMaskDetector-master\test'

train_set = train_datagen.flow_from_directory(
        train,
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        test,
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

model_saved=model.fit_generator(
        train_set,
        epochs=10,
        validation_data=test_set
        )

model.save('FaceMask_model.h5', model_saved)

N = 10
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), model_saved.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), model_saved.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), model_saved.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), model_saved.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss or Accuracy")
plt.legend(loc="upper right")

