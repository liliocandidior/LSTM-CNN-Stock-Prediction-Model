import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow import keras
import matplotlib.pyplot as plt
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import mplcursors
import pandas as pd
# sys.path.insert(0, 'data')
# sys.path.insert(0, 'utils')
sys.path.append('../')
from data import data_acquire
from utils import plot_util
from result import result_analysis
# from plot_util import plotResult
import math
import numpy as np
from sklearn.metrics import mean_squared_error


API_KEY = 'R680A7OABBQ58NL3'
TICKER = 'AAPL'
def cnn(API_KEY, TICKER):
    (data, x_data, y_data, x_train, y_train, x_test, y_test, scaler) = data_acquire.get_data(API_KEY, TICKER)

    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    opt = keras.optimizers.legacy.Adam(learning_rate=0.001)
    import time
    start = time.time()

    model.compile(loss='mean_absolute_error', optimizer=opt)
    model.fit(x_train, y_train, epochs= 30, validation_data=(x_test, y_test), batch_size=64, verbose=1)

    end = time.time()
    print(end - start)

    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    result_analysis.analysis(train_predict, y_train, test_predict, y_test)

    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)


    plot_util.plotResult(data, x_data, model, scaler, train_predict, test_predict, 20, x_train.shape[0])
