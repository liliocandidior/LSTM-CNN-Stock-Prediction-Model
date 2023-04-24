from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, RepeatVector
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0, 'data')
from data_acquire import get_data
import numpy as np
import math

API_KEY = 'R680A7OABBQ58NL3'
TICKER = 'AAPL'

(x_data, y_data, x_train, y_train, x_test, y_test, scaler) = get_data(API_KEY, TICKER)

encoder = Sequential()
encoder.add(Conv1D(32, 1, activation='tanh', padding='same', input_shape=(1,1)))
encoder.add(MaxPooling1D(pool_size=1, padding='same'))
encoder.add(LSTM(128, activation='tanh', return_sequences=False))
encoder.add(Dense(1))

# create the decoder model
decoder = Sequential()
decoder.add(RepeatVector(encoder.output.shape[1]))
decoder.add(Conv1D(32, 1, activation='tanh', padding='same'))
decoder.add(MaxPooling1D(pool_size=1, padding='same'))
decoder.add(LSTM(128, activation='tanh', return_sequences=True))
decoder.add(Dense(1, activation='linear'))

seq2seq_model = Sequential([encoder, decoder])

print(seq2seq_model.summary())
opt = keras.optimizers.Adam(learning_rate=0.001)

import time
start = time.time()

seq2seq_model.compile(loss='mean_absolute_error',optimizer=opt)
seq2seq_model.fit(x_train, y_train, epochs=100, validation_data=(x_test,y_test), batch_size=64, verbose=1)

end = time.time()
print('Time: '+str(end - start))


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100

train_predict = seq2seq_model.predict(x_train)
train_predict=scaler.inverse_transform(train_predict[:, 0])
x_train = scaler.inverse_transform(x_train)

test_predict = seq2seq_model.predict(x_test)
test_predict=scaler.inverse_transform(test_predict[:, 0])
x_test = scaler.inverse_transform(x_test)

acc = calculate_accuracy(x_test, test_predict)

mse = math.sqrt(mean_squared_error(y_train,train_predict))
mse2 = math.sqrt(mean_squared_error(y_test,test_predict))

print(f'######################## Min Squared error training is {mse} ########################')
print(f'######################## Min Squared error testing is {mse2} ########################')
print(f'######################## Accuracy of training is {acc} ########################')

plt.plot(train_predict, c='red', label = 'Prediction')
plt.plot(x_train, c='black', label = 'Actual')
plt.legend()
plt.show()