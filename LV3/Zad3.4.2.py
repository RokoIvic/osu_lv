import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("LV3/data_C02_emission.csv")

# a)
plt.figure()
data["CO2 Emissions (g/km)"].plot(kind="hist", bins=20, color="blue", edgecolor="black")

# b)
data.plot.scatter(x = 'Fuel Consumption City (L/100km)', y = 'CO2 Emissions (g/km)', c = "Fuel Type" , cmap = "hot" , s = 50, edgecolor = "black", alpha = 0.7)

# c)
data.boxplot ( column = [ 'Fuel Consumption Hwy (L/100km)' ] , by = 'Fuel Type')

# d)
data.groupby("Fuel Type").size().plot(kind="bar", color="green", edgecolor="black")

# e)
data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot(kind="bar", color="green", edgecolor="black")

plt.show()

