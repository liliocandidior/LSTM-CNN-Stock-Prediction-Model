import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, TimeDistributed, Conv1D,MaxPooling1D
from tensorflow import keras
import matplotlib.pyplot as plt
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import mplcursors
import pandas as pd
sys.path.insert(0, 'data')
sys.path.insert(0, 'utils')
from data_acquire import get_data
from plot_util import plotResult
import math
import numpy as np
from sklearn.metrics import mean_squared_error


API_KEY = 'R680A7OABBQ58NL3'
TICKER = 'AAPL'

(data, x_data, y_data, x_train, y_train, x_test, y_test, scaler) = get_data(API_KEY, TICKER)

model = Sequential()
model.add(Conv1D(32,1,activation='tanh',padding='same', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(MaxPooling1D(pool_size=1, padding='same'))
model.add(LSTM(64, activation='tanh', return_sequences = True))
model.add(LSTM(64, activation='tanh', return_sequences = True))
model.add(LSTM(64, activation='tanh', return_sequences = True))
model.add(LSTM(64, activation='tanh', return_sequences = True))
model.add(LSTM(64, activation='tanh', return_sequences = True))
model.add(LSTM(64, activation='tanh'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.001)
import time
start = time.time()

model.compile(loss='mean_absolute_error',optimizer=opt)
model.fit(x_train, y_train, epochs=20, validation_data=(x_test,y_test), batch_size=64, verbose=1)

end = time.time()
print(end - start)

train_predict = model.predict(x_train)
train_predict=scaler.inverse_transform(train_predict)

test_predict = model.predict(x_test)
test_predict=scaler.inverse_transform(test_predict)

mse = math.sqrt(mean_squared_error(y_train,train_predict))
mse2 = math.sqrt(mean_squared_error(y_test,test_predict))

print(f'######################## Min Squared error training is {mse} ########################')
print(f'######################## Min Squared error testing is {mse2} ########################')


plotResult(data, x_data, model, scaler, train_predict, test_predict, 20, x_train.shape[0])