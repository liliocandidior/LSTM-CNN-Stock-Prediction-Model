from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

'''
Sample Args passed into get_data() function
API_KEY: Fixed, I generated it via alpha_vantage platform
TICKER: Variable, whichever stock we want to predict

    API_KEY = 'R680A7OABBQ58NL3'
    TICKER = 'AAPL'
'''


def create_dataset(data, n_past):
    x = []
    y = []
    for i in range(len(data) - n_past - 1):
        x.append(data[i:(i + n_past)])
        y.append(data[i + n_past, 0])
    return np.array(x), np.array(y)


def get_data(api_key, ticker):
    ts = TimeSeries(api_key, output_format='pandas')

    # Retrieve data from timeseries
    data = ts.get_daily_adjusted(ticker, 'full')[0]
    # Use close price & volumn as the feature space
    data = data.iloc[:, 3]

    # Sort by ascending timestamps to match intuition
    data = data.astype(float).sort_index(ascending=True, axis=0).tail(3000)

    # Standarize the data, MinMaxScaler picked for stock price prediction (from online resources)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1,1))

    # Split the data into x_data & y_data for training, testing, and prediction purpose
    x_data, y_data = create_dataset(data_scaled, 20)

    # Preserve the 20-day shape of X data
    x_data = x_data.reshape(x_data.shape[0],x_data.shape[1],1)


    # Shuffle = false since we cannot mess up with the sequence for timeseries data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.05, shuffle=False)
    print(f'\n######################## Data for {ticker} scaled and partitioned complete ########################')
    print(
         f'Sample Number: {str(x_data.shape[0])}; Train Sample Number: {str(x_train.shape[0])}; Test Sample Number: {str(x_test.shape[0])}')
    print(f'###############################################################################################\n')

    return data, x_data, y_data, x_train, y_train, x_test, y_test, scaler


