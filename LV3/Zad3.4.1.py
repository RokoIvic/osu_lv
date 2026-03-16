import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("LV3/data_C02_emission.csv")

print(data.head())

# a)
print("Broj mjerenja je: ", data.shape[0])
print(data.info())
print("Broj izostalih vrijednosti je: ", data.isnull().sum().sum())
new_data = data.drop_duplicates()
print("Broj dupliciranih vrijednosti je: ", data.shape[0] - new_data.shape[0])
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data["Make"] = data["Make"].astype("category")
data["Model"] = data["Model"].astype("category") 
data["Vehicle Class"] = data["Vehicle Class"].astype("category")
data["Transmission"] = data["Transmission"].astype("category")  
data["Fuel Type"] = data["Fuel Type"].astype("category")   

# b)
# Fuel Consumption City (L/100km)
dataFilteredByFuelConsumptionCity = data.sort_values(by="Fuel Consumption City (L/100km)")

print("Tri s najmanjom potrošnjom goriva u gradu: \n", dataFilteredByFuelConsumptionCity.iloc[:3][['Make', 'Model', 'Fuel Consumption City (L/100km)']])

dataFilteredByFuelConsumptionCity = data.sort_values(by="Fuel Consumption City (L/100km)", ascending=False)

print("Tri s najvećom potrošnjom goriva u gradu: \n", dataFilteredByFuelConsumptionCity.iloc[:3][['Make', 'Model', 'Fuel Consumption City (L/100km)']])

# c)
# Engine Size (L)
dataFilteredByEngineSize = (data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)
print("Broj vozila s motorom između 2.5L i 3.5L je: ", dataFilteredByEngineSize.sum())

print("prosječna C02 emisija plinova za ova vozila je: ", data[dataFilteredByEngineSize]["CO2 Emissions (g/km)"].mean())

# d)
print("Broj Audi vozila je: ", data[data["Make"] == "Audi"].shape[0])
print("Prosječna emisija CO2 za Audi vozila s 4 cilindra je: ", data[data["Make"] == "Audi"][data["Cylinders"] == 4]["CO2 Emissions (g/km)"].mean())

# e)
types_of_cylinders = data["Cylinders"].unique()
for cylinder in types_of_cylinders:
    print("Broj vozila s ", cylinder, " cilindara je: ", data[data["Cylinders"] == cylinder].shape[0])
    print("Prosječna emisija CO2 za vozila s ", cylinder, " cilindara je: ", data[data["Cylinders"] == cylinder]["CO2 Emissions (g/km)"].mean(), "g/kg")

# f)
print("prosječna gradska potrošnja kod vozila koja koriste dizel: ", data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].mean(), "L/100km", "a medijala je: ", data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].median(), "L/100km")
print("prosječna gradska potrošnja kod vozila koja koriste benzin: ", data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].mean(), "L/100km", "a medijala je: ", data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].median(), "L/100km")

# g)
filtered_data = (data["Cylinders"] == 4) & (data["Fuel Type"] == "D")
print("Vozilo s 4 cilindra koje koristi dizelski motor ima najveću gradsku potrošnju goriva je:", data[filtered_data].sort_values(by="Fuel Consumption City (L/100km)", ascending=False).iloc[0][['Make', 'Model', 'Fuel Consumption City (L/100km)']])

# h)
print("Broj vozil as ručnim mjenjačem je: ", data[data["Transmission"].str.startswith("M")].shape[0])

# i)
print("Korelacija između numeričkih vrijednosti je: ", data.corr(numeric_only=True))