import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import mplcursors

def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    print ('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
    print ('Data point:', x[ind[0]], y[ind[0]])

def plotResult(data, x_data, model, scaler, train_predict, test_predict, n_past, training_size):
    df_for_training_plt = data.reset_index().iloc[:,1].tolist()

    train_predict_plt = (n_past)*[np.NaN] + list(train_predict[:,0])
    test_predict_plt = (n_past+training_size)*[np.NaN] + list(test_predict[:,0])

    # # Determine past and future data to plot and plug back into the date dataframe
    us_bd = CustomBusinessDay(calendar = USFederalHolidayCalendar())
    to_past = 60
    to_future = 14
    forecast_period_dates = pd.date_range(data.index[-1], periods=to_future, freq=us_bd).tolist()
    past_period_dates = pd.date_range(data.index[-to_past], periods=to_past, freq=us_bd).tolist()

    prediction = []
    current_batch = x_data[-1, :, :].reshape(1,20,1)
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