from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
def analysis(y_pred_train, y_actual_train, y_pred_test, y_actual_test):
    r2_train = r2_score(y_actual_train, y_pred_train)
    r2_test = r2_score(y_actual_test, y_pred_test)

    mse_train = mean_squared_error(y_actual_train, y_pred_train)
    mse_test = mean_squared_error(y_actual_test, y_pred_test)

    print(f'######################## Min Squared error training is {mse_train} ########################')
    print(f'######################## Min Squared error testing is {mse_test} ########################')
    print(f'######################## R2 score of Training Dataset: {r2_train} ########################')
    print(f'######################## R2 score of Testing Dataset: {r2_test} ########################')
