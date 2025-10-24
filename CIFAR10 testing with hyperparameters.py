
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers, initializers
from tabulate import tabulate
import time
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
results = []    # For Store results
def run_test(name, initializer, regularizer=None, use_dropout=False):
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(512, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer))
    if use_dropout: model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(Dense(10, activation='softmax', kernel_initializer=initializer))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=0)
    end_time = time.time()
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    overfit = "Yes" if (train_acc - val_acc) > 0.1 else "No"
    total_time = end_time - start_time
    results.append({                                                                
        "Model": name,
        "Train Acc (%)": f"{train_acc*100:.2f}",
        "Val Acc (%)": f"{val_acc*100:.2f}",
        "Test Acc (%)": f"{test_acc*100:.2f}",
        "Overfit?": overfit,
        "Train Time (s)": f"{total_time:.2f}"
    })
run_test("0) Baseline (Uniform)", initializer='uniform')
run_test("a) Xavier Init", initializer=initializers.glorot_uniform())
run_test("b) Kaiming Init", initializer=initializers.he_uniform())
run_test("c) Dropout (0.3)", initializer='uniform', use_dropout=True)
run_test("d) L2 Regularization", initializer='uniform', regularizer=regularizers.l2(0.001))
print("\nFinal Comparison Table:\n")
print(tabulate(results, headers="keys", tablefmt="simple"))
