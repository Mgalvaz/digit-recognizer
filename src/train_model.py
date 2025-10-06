"""
Train two different CNN models on the MNIST dataset, evaluate them,
and save the models and their training history to the 'models' directory.
"""

import json, os
from keras.losses import CategoricalCrossentropy
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

def save_history(history, test_loss, test_acc, filename):
    data = {
        'history': history.history,
        'test': {'test_loss': test_loss, 'test_accuracy': test_acc}
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

# Import the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Modify the images so that layers can "understand" the data
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32')
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32')

# Labels one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create both CNN models
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

# Train the model
model1.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model2.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history1 = model1.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)
history2 = model2.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)

# Save the model
os.makedirs('models', exist_ok=True)
model1.save('models/mnist_cnn1.keras')
model2.save('models/mnist_cnn2.keras')

# Save info
save_history(history1, test_loss1, test_acc1, 'models/history1.json')
save_history(history2, test_loss2, test_acc2, 'models/history2.json')