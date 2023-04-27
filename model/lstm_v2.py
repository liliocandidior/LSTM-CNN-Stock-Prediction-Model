import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, TimeDistributed, Conv1D,MaxPooling1D
from tensorflow.keras.models import load_model
from tensorflow import keras
import matplotlib.pyplot as plt
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
sys.path.insert(0, 'data')
sys.path.insert(0, 'utils')
from data_acquire import get_data
from plot_util import plotResult
import math
import numpy as np


API_KEY = 'R680A7OABBQ58NL3'
TICKER = 'AAPL'

def lstm(API_KEY, TICKER, MODE):
    (data, x_data, y_data, x_train, y_train, x_test, y_test, scaler) = get_data(API_KEY, TICKER)

    if MODE == "TRAINING":
        model = Sequential()
        model.add(Conv1D(64,1,activation='tanh',padding='same', input_shape=(x_train.shape[1],x_train.shape[2])))
        model.add(MaxPooling1D(pool_size=1, padding='same'))
        model.add(LSTM(256, activation='tanh', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, activation='tanh', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, activation='tanh'))
        model.add(Dense(1))
        opt = keras.optimizers.Adam(learning_rate=0.001)
        import time
        start = time.time()

        model.compile(loss='mean_absolute_error',optimizer=opt)
        model.fit(x_train, y_train, epochs=100, validation_data=(x_test,y_test), batch_size=64, verbose=1)
        model.save(f'Models/{TICKER}_LSTM.h5')

    else:
        model = load_model(f'Models/{TICKER}_LSTM.h5')

    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    analysis(train_predict, y_train, test_predict, y_test)

    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

mse = math.sqrt(mean_squared_error(y_train,train_predict))
mse2 = math.sqrt(mean_squared_error(y_test,test_predict))

print(f'######################## Min Squared error training is {mse} ########################')
print(f'######################## Min Squared error testing is {mse2} ########################')


plotResult(data, x_data, model, scaler, train_predict, test_predict, 20, x_train.shape[0])