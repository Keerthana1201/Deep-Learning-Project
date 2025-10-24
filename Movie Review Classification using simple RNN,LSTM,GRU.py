
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, SimpleRNN
num_words = 10000
max_length = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)
RNN_Model = Sequential()
RNN_Model.add(Embedding(input_dim=num_words, output_dim=128, input_length=max_length))
RNN_Model.add(SimpleRNN(64))
RNN_Model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

RNN_Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
LSTM_Model = Sequential()
LSTM_Model.add(Embedding(input_dim=num_words, output_dim=128, input_length=max_length))
LSTM_Model.add(LSTM(64))
LSTM_Model.add(Dense(1, activation='sigmoid'))

LSTM_Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
GRU_Model = Sequential()
GRU_Model.add(Embedding(input_dim=num_words, output_dim=128, input_length=max_length))
GRU_Model.add(GRU(64))
GRU_Model.add(Dense(1, activation='sigmoid'))

GRU_Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
LSTM_history = LSTM_Model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64)
GRU_history  = GRU_Model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64)
RNN_history  = RNN_Model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64)
from tabulate import tabulate

# Evaluate models
LSTM_loss, LSTM_acc = LSTM_Model.evaluate(X_test, y_test)
GRU_loss, GRU_acc   = GRU_Model.evaluate(X_test, y_test)
RNN_loss, RNN_acc   = RNN_Model.evaluate(X_test, y_test)

# Prepare data for table
results = [
    ["LSTM", f"{round(LSTM_acc * 100, 2)}%", round(LSTM_loss, 4)],
    ["GRU",  f"{round(GRU_acc * 100, 2)}%", round(GRU_loss, 4)],
    ["RNN",  f"{round(RNN_acc * 100, 2)}%", round(RNN_loss, 4)]
]

# Print table
print(tabulate(results, headers=["Model", "Accuracy", "Loss"], tablefmt="grid"))
plt.plot(LSTM_history.history['accuracy'], label='LSTM Training Accuracy')
plt.plot(LSTM_history.history['val_accuracy'], label='LSTM Validation Accuracy')

plt.plot(GRU_history.history['accuracy'], label='GRU Training Accuracy')
plt.plot(GRU_history.history['val_accuracy'], label='GRU Validation Accuracy')

plt.plot(RNN_history.history['accuracy'], label='RNN Training Accuracy')
plt.plot(RNN_history.history['val_accuracy'], label='RNN Validation Accuracy')


plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.show()
plt.plot(LSTM_history.history['loss'], label='LSTM Training Loss')
plt.plot(LSTM_history.history['val_loss'], label='LSTM Validation Loss')

plt.plot(GRU_history.history['loss'], label='GRU Training Loss')
plt.plot(GRU_history.history['val_loss'], label='GRU Validation Loss')

plt.plot(RNN_history.history['loss'], label='RNN Training Loss')
plt.plot(RNN_history.history['val_loss'], label='RNN Validation Loss')


plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()
