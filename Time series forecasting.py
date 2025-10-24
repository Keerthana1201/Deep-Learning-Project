import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization

df = pd.read_csv('Nifty.csv').iloc[::-1]
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date', drop=True)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

def create_lagged_features(data, look_back=30):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:i + look_back])
        y.append(data[i + look_back, df.columns.get_loc('Close')])
    return np.array(x), np.array(y)

look_back=5
data_x, data_y = create_lagged_features(scaled_df.values, look_back)
train_size = 0.8
train_index = int(len(data_x) * train_size)
split_date = df.index[int(look_back + train_size * (len(df) - look_back))]
x_train, x_test = data_x[:train_index], data_x[train_index:]
y_train, y_test = data_y[:train_index], data_y[train_index:]

model = Sequential([
    LSTM(64, input_shape=(look_back, 4)),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

history = model.fit(x_train, y_train, epochs=50, verbose=1, callbacks=[early_stop], shuffle=False)

y_test_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_test_pred)
print(f'Mean Absolute Error: {mae}')
y_test_pred_inv = scaler.inverse_transform(np.concatenate((np.zeros((y_test_pred.shape[0], 3)), y_test_pred), axis=1))[:, -1]
y_test_inv = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 3)), y_test.reshape(-1, 1)), axis=1))[:, -1]
test_dates = df.index[train_index + look_back:]

plt.figure(figsize=(15, 8))
plt.plot(df.index, df['Close'], label='All Data', color='grey', alpha=0.5)
plt.plot(df.index[:train_index + look_back], df['Close'][:train_index + look_back], label='Train', color='blue')
plt.plot(test_dates, y_test_inv, label='True Values', color='green', linestyle='--')
plt.plot(test_dates, y_test_pred_inv, label='Predicted Values', color='red', linestyle='--')
plt.axvline(x=split_date, color='black', linestyle='--', label='Split Date')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.title("Train Data, True Values, and Predictions")
plt.show()
