
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten,Dense,MaxPooling2D,Conv2D
from tensorflow.keras import layers,models,regularizers
import numpy as np
import matplotlib.pyplot as plt

(xtrain,ytrain),(xtest,ytest) = mnist.load_data()
xtrain,xtest = xtrain/255.0,xtest/255.0
ytrain,ytest = to_categorical(ytrain),to_categorical(ytest)

model = models.Sequential([
   
    Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(),
    Conv2D(64,3,activation='relu'),
    MaxPooling2D(),
    Conv2D(64,3,activation='relu'),
    Flatten(),
    Dense(10,activation='softmax'),
])
model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.history = model.fit(xtrain,ytrain,epochs = 5,batch_size = 64,validation_data = (xtest,ytest))
base_test_loss,base_test_acc = model.evaluate(xtest,ytest)
print("accuracy = ",base_test_acc)
print("loss = ",base_test_loss)
