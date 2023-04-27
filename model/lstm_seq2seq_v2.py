from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Dot, Input, Attention, Concatenate
from keras import backend as K
from tensorflow import keras
import matplotlib.pyplot as plt
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import mplcursors
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0, 'data')
sys.path.insert(0, 'utils')
from data_acquire import get_data
from plot_util import plotResult
import numpy as np
import math

API_KEY = 'R680A7OABBQ58NL3'
TICKER = 'AAPL'

(data, x_data, y_data, x_train, y_train, x_test, y_test, scaler) = get_data(API_KEY, TICKER)
n_steps = x_train.shape[1]
num_features = x_train.shape[2]

# define the model
# define the model
model = Sequential()

# add the encoder layers
encoder_inputs = Input(shape=(n_steps, num_features))
encoder = LSTM(64, activation='relu', return_sequences=True)(encoder_inputs)
encoder = LSTM(64, activation='relu', return_sequences=True)(encoder)
encoder = Dropout(0.1)(encoder)

# add the decoder layers
decoder = LSTM(64, activation='relu', return_sequences=True)(encoder)
decoder = LSTM(64, activation='relu', return_sequences=False)(decoder)
decoder = Dropout(0.1)(decoder)

# add the output layer
outputs = Dense(1, activation='linear')(decoder)

# compile the model
model = Model(inputs=encoder_inputs, outputs=outputs)

print(model.summary())
opt = keras.optimizers.Adam(learning_rate=0.001)

import time
start = time.time()

model.compile(loss='mse',optimizer=opt)
model.fit(x_train, y_train, epochs=20, validation_data=(x_test,y_test), batch_size=64, verbose=1)

end = time.time()
print('Time: '+str(end - start))

train_predict = model.predict(x_train)
train_predict=scaler.inverse_transform(train_predict)

print(train_predict)

test_predict = model.predict(x_test)
test_predict=scaler.inverse_transform(test_predict)

mse = math.sqrt(mean_squared_error(y_train,train_predict))
mse2 = math.sqrt(mean_squared_error(y_test,test_predict))

print(f'######################## Min Squared error training is {mse} ########################')
print(f'######################## Min Squared error testing is {mse2} ########################')

plotResult(data, x_data, model, scaler, train_predict, test_predict, 20, x_train.shape[0])