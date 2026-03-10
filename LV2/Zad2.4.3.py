import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("LV2/road.jpg")
img = img[:, :, 0].copy()   

print(img.shape)

fig, ax = plt.subplots(5)

ax[0].imshow(img, cmap="gray")   

imgBrightened = img.copy()

for i in range(imgBrightened.shape[0]):
    for j in range(imgBrightened.shape[1]):
        if imgBrightened[i, j] < 128:
            imgBrightened[i, j] = imgBrightened[i, j] + 100


ax[1].imshow(imgBrightened, cmap="gray")

imgCropped = img[:, int(img.shape[1]*0.25):int(img.shape[1]*0.75):2].copy()
ax[2].imshow(imgCropped, cmap="gray")

imgRotated = np.rot90(img).copy()
ax[3].imshow(imgRotated, cmap="gray")

imgMirrored = np.flip(img, axis=1).copy()
ax[4].imshow(imgMirrored, cmap="gray")

plt.show()

