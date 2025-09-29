import json, os
from keras.losses import CategoricalCrossentropy
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
#Import the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Modify the images so that layers can "understand" the data
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32')
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32')

#Labels one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Create the CNN model

model1 = models.Sequential([
    layers.Input((28, 28, 1)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='linear')
    ])

model2 = models.Sequential([
    layers.Input((28,28,1)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'),
    layers.MaxPooling2D(strides=(2,2), padding='valid'),
    layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.MaxPooling2D(strides=(2,2), padding='valid'),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='linear')
])

#Train the model
model1.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model2.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history1 = model1.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)
history2 = model2.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

#Evaluate the model
test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)

#Save the model
os.makedirs('models', exist_ok=True)
model1.save('models/mnist_cnn1.keras')
model2.save('models/mnist_cnn2.keras')

#Save info on model 1
history1_data = {
    'history': history1.history,
    'test': {
        'test_loss': test_loss1,
        'test_accuracy': test_acc1
    }
}
with open('models/history1.json', 'w') as f:
    json.dump(history1_data, f)
#Save info on model 2
history2_data = {
    'history': history2.history,
    'test': {
        'test_loss': test_loss2,
        'test_accuracy': test_acc2
    }
}
with open('models/history2.json', 'w') as f:
    json.dump(history2_data, f)
