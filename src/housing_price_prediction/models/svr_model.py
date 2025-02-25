from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def train_svr(x_train, y_train, x_test, y_test):
    model = SVR()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return predictions, mse, r2
