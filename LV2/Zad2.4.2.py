import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("LV2/data.csv", delimiter=",", skip_header=1)

# a)
print("Broj osoba je: ", data.shape[0])

# b)
heights = np.array(data[:,1])
weights = np.array(data[:,2])

plt.scatter(heights, weights)
plt.xlabel('Visina')
plt.ylabel('Težina')
plt.title('Zavisnost težine od visine')
plt.show()

# c)
heights = np.array(data[::40,1])
weights = np.array(data[::40,2])

plt.scatter(heights, weights)
plt.xlabel('Visina svake 40. osobe')
plt.ylabel('Težina svake 40. osobe')   
plt.title('Zavisnost težine od visine')
plt.show()

# d)
heights = np.array(data[::,1])
print("Minimalna visina je: ", np.min(heights))
print("Maksimalna visina je: ", np.max(heights))
print("Prosječna visina je: ", np.mean(heights))

# e)
menIndexes = (data[:,0] == 1)
womenIndexes = (data[:,0] == 0)

print("Minimalna visina muškaraca je: ", np.min(data[menIndexes,1]))
print("Maksimalna visina muškaraca je: ", np.max(data[menIndexes,1]))
print("Prosječna visina muškaraca je: ", np.mean(data[menIndexes,1]))

print("Minimalna visina žena je: ", np.min(data[womenIndexes,1]))
print("Maksimalna visina žena je: ", np.max(data[womenIndexes,1]))
print("Prosječna visina žena je: ", np.mean(data[womenIndexes,1]))


