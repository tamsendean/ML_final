import numpy as np
import pandas as pd
import pylab
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

if __name__ == '__main__':
    data = pd.read_csv("end_times.csv")

    X_set = data[['Day', 'Ride Time (secs)', 'Average Heart Rate (bpm)']]
    y_set = data['Average Speed (mph)']

    regr = linear_model.LinearRegression()
    regr.fit(X_set, y_set)

    predicted = regr.predict([[200, 2500, 120]])

    print("Predicted Speed: ", predicted)

    