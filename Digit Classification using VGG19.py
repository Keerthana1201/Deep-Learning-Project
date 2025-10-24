
import numpy as np
from tensorflow.image import resize
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = np.expand_dims(xtrain, -1)
xtest = np.expand_dims(xtest, -1)
xtrain = np.repeat(xtrain, 3, axis=-1)
xtest = np.repeat(xtest, 3, axis=-1)
xtrain = resize(xtrain, (32,32))
xtest = resize(xtest, (32,32))
xtrain, xtest = xtrain / 255.0, xtest / 255.0
ytrain, ytest = to_categorical(ytrain), to_categorical(ytest)

vgg = VGG19(weights='imagenet',input_shape=(32,32,3), include_top=False)
for layer in vgg.layers:
    layer.trainable = False

layer_2 = Flatten()(vgg.output)
layer_3 = Dense(512, activation='relu')(layer_2)
output = Dense(10, activation='sigmoid')(layer_3)

model = Model(inputs=vgg.input, outputs = output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(xtrain, ytrain, batch_size=32, epochs=5, validation_data=(xtest, ytest))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
plt.tight_layout()
fig.suptitle('Fixed Feature Extractor', fontsize=20)
fig.subplots_adjust(top=0.85)
plt.show()

vgg = VGG19(weights='imagenet',input_shape=(32,32,3), include_top=False)

for layer in vgg.layers:
    layer.trainable = False

for layer in vgg.layers[-4:]:
    layer.trainable = True
    
layer_2 = Flatten()(vgg.output)
layer_3 = Dense(512, activation='relu')(layer_2)
output = Dense(10, activation='softmax')(layer_3)
model = Model(inputs=vgg.input, outputs = output)
    
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model = Model(inputs=vgg.input, outputs = output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(xtrain, ytrain, batch_size=32, epochs=5, validation_data=(xtest, ytest))
loss, accuracy = model.evaluate(xtest, ytest)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
plt.tight_layout()
fig.suptitle('Fine Tuned Model', fontsize=20)
fig.subplots_adjust(top=0.85)
plt.show()
