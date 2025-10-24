import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
def create_model(hidden_units=None, activation=None):
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(hidden_units[0], activation=activation),
        Dense(hidden_units[1], activation=activation),
        Dense(hidden_units[2], activation=activation),
        Dense(10, activation='softmax')
    ])
    return model
hidden_units_list = [(512, 256, 128), (256, 128, 64), (1024, 512, 256)]
activation_list = ['relu', 'tanh', 'sigmoid']
result_dict = {}
counter = 1
for hidden_units in hidden_units_list:
    for activation in activation_list:
        model = create_model(hidden_units=hidden_units, activation=activation)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=0)
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        model_info = {
            'Hidden units': hidden_units,
            'Activation': activation,
            'Test accuracy': round(test_acc * 100, 4)
        }
        result_dict[counter] = model_info
        counter += 1
for i in range(1, 10):
    print(result_dict[i])
max_result = max(result_dict.values(), key=lambda x: x['Test accuracy'])
print("\nMaximum accuracy:", max_result)
best_hidden_units = (256, 128, 64)
best_model = create_model(hidden_units=best_hidden_units, activation='relu')
best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
best_model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
indices = [10, 20, 50]
images = x_test[indices]
labels = y_test[indices]
preds = best_model.predict(images)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 3))
for i, img in enumerate(images):
    pred = np.argmax(preds[i])
    true = np.argmax(labels[i])
    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Pred: {class_names[pred]}\nTrue: {class_names[true]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
for i, probs in enumerate(preds):
    print(f"\nImage {i+1} predictions:")
    for cls, prob in zip(class_names, probs):
        print(f" {cls}: {prob:.4f}")
    print(f"--> Predicted class: {class_names[np.argmax(probs)]}")
