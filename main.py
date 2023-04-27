import sys
# sys.path.insert(0, 'model')
sys.path.append('../')
from model import lstm_v2
from model import cnn
from model import cnn_s2s

API_KEY = 'R680A7OABBQ58NL3'
def main(argv):
    global API_KEY
    if len(argv) < 4:
        print('Usage: python main.py <stock (AAPL, TSLA, etc..)> <model (lstm or seq2seq or cnn)> <mode (TRAINING or TESTING)>')
        sys.exit(1)
    # Update Mode and Stock
    TICKER = argv[-3]
    MODEL = argv[-2]
    MODE = argv[-1]

    if MODEL == 'lstm':
        lstm(API_KEY, TICKER, MODE)
    if MODEL == 'seq2seq':
        pass
    if MODEL == 'cnn':
        cnn.cnn(API_KEY, TICKER)
    if MODEL == 'cnn_s2s':
        cnn_s2s.cnn_s2s(API_KEY, TICKER)

    # for TICKER in TICKERS:
    #     print(f'\n######################## Beginning to process on {TICKER} with Mode {MODE} ########################')
    #     stock_predict()

if __name__ == '__main__':
    main(sys.argv)