import sys
sys.path.insert(0, 'model')
from lstm_v2 import lstm

API_KEY = 'R680A7OABBQ58NL3'
def main(argv):
    global API_KEY
    if len(argv) < 3:
        print('Usage: python main.py <stock (AAPL, TSLA, etc..)> <model (lstm or seq2seq or cnn)>')
        sys.exit(1)
    # Update Mode and Stock
    TICKER = argv[-2]
    MODEL = argv[-1]

    if MODEL == 'lstm':
        lstm(API_KEY, TICKER)
    if MODEL == 'seq2seq':
        pass
    if MODEL == 'cnn':
        pass

    # for TICKER in TICKERS:
    #     print(f'\n######################## Beginning to process on {TICKER} with Mode {MODE} ########################')
    #     stock_predict()

if __name__ == '__main__':
    main(sys.argv)