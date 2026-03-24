from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# a)
data = pd.read_csv("LV4/data_C02_emission.csv")

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()

X = data[["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)"]]

X = np.hstack([X.values, X_encoded])

y = data["CO2 Emissions (g/km)"]

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train , y_train)  

y_test_p = linearModel.predict(X_test)

error = np.abs(y_test - y_test_p)

max_error_index = error.idxmax()

print("Model auta", data["Model"][max_error_index])

print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Mean squared error:", metrics.mean_squared_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Root mean squared error:", metrics.root_mean_squared_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Mean absolute percentage error:", metrics.mean_absolute_percentage_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))



