import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("LV7\imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()


# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# 1
print("Broj boja je: ", len(np.unique(img_array, axis=0)))

# 2
model = KMeans()
model.fit(img_array_aprox[:, :])
model_predictions = model.predict(img_array_aprox)

img_K_8 = img_array.copy()

# 3
for i in range(len(img_array_aprox)):
    img_K_8[i, :] = model.cluster_centers_[model_predictions[i]]

img_aprox = np.reshape(img_K_8, (w, h, d))
plt.figure()
plt.title("Aproksimacija slike K=8")
plt.imshow(img_aprox)
plt.tight_layout()

# 4
model_2 = KMeans(n_clusters=16)
model_2.fit(img_array_aprox[:, :])
model_2_predictions = model_2.predict(img_array_aprox)

img_K_16 = img_array.copy()

for i in range(len(img_array_aprox)):
    img_K_16[i, :] = model_2.cluster_centers_[model_2_predictions[i]]

img_aprox_16 = np.reshape(img_K_16, (w, h, d))
plt.figure()
plt.title("Aproksimacija slike K=16")
plt.imshow(img_aprox_16)
plt.tight_layout()


# 5
for i in range(5):
    img_loop = Image.imread("LV7\imgs\\test_" + str(i + 2) + ".jpg")
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img_loop)
    plt.tight_layout()
    img_loop = img_loop.astype(np.float64) / 255
    w,h,d = img_loop.shape
    img_array = np.reshape(img_loop, (w*h, d))

    model = KMeans(n_clusters=8)
    model.fit(img_array[:, :])
    model_predictions = model.predict(img_array)

    img_copy = img_array.copy() 

    for j in range(len(img_copy)):
        img_copy[j, :] = model.cluster_centers_[model_predictions[j]]

    img_aprox = np.reshape(img_copy, (w, h, d))
    plt.figure()
    plt.title("Aproksimacija slike " + str(i + 2) + " K=8")
    plt.imshow(img_aprox)
    plt.tight_layout()    

# 6
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(img_array[:, :])
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# 7
img = Image.imread("LV7\imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()


# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

model = KMeans()
model.fit(img_array_aprox[:, :])
model_predictions = model.predict(img_array_aprox)

img_K_8 = img_array.copy()

for i in range(8):
    for j in range(len(img_array_aprox)):
        if model_predictions[j] == i:
            img_K_8[j, :] = model.cluster_centers_[model_predictions[j]]
        else:
            img_K_8[j, :] = [1, 1, 1]

    img_aprox = np.reshape(img_K_8, (w, h, d))
    plt.figure()
    plt.title("Feature " + str(i))
    plt.imshow(img_aprox)
    plt.tight_layout()




plt.show()
