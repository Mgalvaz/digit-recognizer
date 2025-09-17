import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import os

#Import the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Modify the images so that layers can "understand" the data
train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32")
test_images = test_images.reshape((-1, 28, 28, 1)).astype("float32")

#Labels one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Create the CNN model
"""
model = models.Sequential([
    layers.Input((28, 28, 1)),
    layers.Rescaling(1. / 255),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='linear')
    ])"""

model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Rescaling(1./255),
    layers.Conv2D(128, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'),
    layers.MaxPooling2D(strides=(2,2), padding='valid'),
    layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.MaxPooling2D(strides=(2,2), padding='valid'),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='linear')
])

#Train the model
model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

#Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

#Save the model
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_cnn.h5")
model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")