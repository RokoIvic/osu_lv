from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# a)
data = pd.read_csv("LV4/data_C02_emission.csv")

X = data[["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)"]]

y = data["CO2 Emissions (g/km)"]

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# b)
plt.scatter(X_train["Engine Size (L)"], y_train, color = "blue", label = "Podaci za treniranje", s=8, alpha=0.5)

plt.scatter(X_test["Engine Size (L)"], y_test, color = "red", label = "Podaci za testiranje", s=8, alpha=0.5)

plt.figure(1)
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()

# c)
scale = StandardScaler()

plt.figure(2)
plt.hist(x = X_train["Engine Size (L)"], bins = 20, color = "blue", label = "Engine Size (L)")
plt.title("Podaci za treniranje")
plt.legend()

X_train_scaled = scale.fit_transform(X_train)

plt.figure(3)
plt.hist(x = X_train_scaled[:, 0], bins = 20, color = "blue", label = "Engine Size (L)")
plt.title("Podaci za treniranje SKALIRANI")
plt.legend()

X_test_scaled = scale.transform(X_test)

# d)
linearModel = lm.LinearRegression()
linearModel.fit(X_train_scaled , y_train)

print("Koeficijenti:", linearModel.coef_)
print("Slobodni član:", linearModel.intercept_)

# e)
y_test_p = linearModel.predict(X_test_scaled)

plt.figure(4)
plt.scatter(y_test, y_test_p, color = "red", alpha = 0.5, s = 10)
plt.title("odnos izmedu stvarnih vrijednosti izlazne velicine i procjene dobivene modelom")
plt.xlabel("y_test")
plt.ylabel("y_test_p")

# f)    
print("\nZa 5 ulzane velicine:")
print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Mean squared error:", metrics.mean_squared_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Root mean squared error:", metrics.root_mean_squared_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Mean absolute percentage error:", metrics.mean_absolute_percentage_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))

# g)
# 4 ulazne velicine
linearModel.fit(X_train_scaled[:, :4] , y_train)
y_test_p = linearModel.predict(X_test_scaled[:, :4])

print("\nZa 4 ulazne velicine:")
print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Mean squared error:", metrics.mean_squared_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Root mean squared error:", metrics.root_mean_squared_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Mean absolute percentage error:", metrics.mean_absolute_percentage_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))

#2 ulazne velicine
linearModel.fit(X_train_scaled[:, :2] , y_train)
y_test_p = linearModel.predict(X_test_scaled[:, :2])

print("\nZa 2 ulazne velicine:")
print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Mean squared error:", metrics.mean_squared_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Root mean squared error:", metrics.root_mean_squared_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))
print("Mean absolute percentage error:", metrics.mean_absolute_percentage_error(y_test, y_test_p, sample_weight=None, multioutput='uniform_average'))

plt.show()


