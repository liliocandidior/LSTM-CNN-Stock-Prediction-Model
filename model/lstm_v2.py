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
from data_acquire import get_data
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

def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    print ('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
    print ('Data point:', x[ind[0]], y[ind[0]])
    
def plotResult(n_past, training_size):
    df_for_training_plt = data.reset_index().iloc[:,1].tolist()

    train_predict_plt = (n_past)*[np.NaN] + list(train_predict[:,0])
    test_predict_plt = (n_past+training_size)*[np.NaN] + list(test_predict[:,0])

    # # Determine past and future data to plot and plug back into the date dataframe
    us_bd = CustomBusinessDay(calendar = USFederalHolidayCalendar())
    to_past = 60
    to_future = 7
    forecast_period_dates = pd.date_range(data.index[-1], periods=to_future, freq=us_bd).tolist()
    past_period_dates = pd.date_range(data.index[-to_past], periods=to_past, freq=us_bd).tolist()

    prediction = []
    current_batch = x_data[-1:]
    for i in range(to_future):
        current_pred = model.predict(current_batch)[0]
        prediction.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
    # Predict past days and inverse scale future predicts to real stock prices
    past_prediction = model.predict(x_data[-to_past:])

    y_pred_past = scaler.inverse_transform(past_prediction)[:,0]
    y_pred_future = scaler.inverse_transform(prediction)[:, 0]

    # Plug back in past and future forecast dates
    future_forecast_dates = []
    for time_i in forecast_period_dates:
        future_forecast_dates.append(time_i.date())
        
    past_forecast_dates = []
    for time_i in past_period_dates:
        past_forecast_dates.append(time_i.date())
        
    df_forecast = pd.DataFrame({'Date': np.array(future_forecast_dates), 'Close': y_pred_future})
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    df_past = pd.DataFrame({'Date': np.array(past_forecast_dates), 'Close': y_pred_past})
    df_past['Date'] = pd.to_datetime(df_past['Date'])

    original = pd.DataFrame({'Date': pd.to_datetime(data.index), 'Close': data})
    original = original.loc[original['Date']>='2023-01-01']

    print(f'######################## Past Predict from {past_forecast_dates[0]} to {past_forecast_dates[-1]} ########################')
    print(f'######################## Future Forecast from {future_forecast_dates[0]} to {future_forecast_dates[-1]} ########################')

    # Plot Past day and future predictions
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
    fig.suptitle('Training Prediction and Future Prediction')
    fig.canvas.callbacks.connect('pick_event', on_pick)
    ax1.plot(df_for_training_plt, label='Original Data')
    ax1.plot(train_predict_plt, label='Training Predict')
    ax1.plot(test_predict_plt, label='Testing Predict')
    ax1.set_title('Plot for Original, Training Predictoin, Testing Prediction')
    ax1.set(xlabel='Date', ylabel='Stock Price')
    ax1.legend()

    ax2.plot(original['Date'], original['Close'], label="Original Data")
    ax2.plot(df_past['Date'], df_past['Close'], label="Past Predcition Data")
    ax2.plot(df_forecast['Date'], df_forecast['Close'], label="Future Prediction Data")
    ax2.set_title(f'Future {to_future} days prediction and Past {to_past} days fitting')
    ax2.set(xlabel='Date', ylabel='Stock Price')
    ax2.legend()
    mplcursors.cursor()
    plt.show()

plotResult(20, x_train.shape[0])