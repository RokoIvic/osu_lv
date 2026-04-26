import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from keras.models import load_model
import cv2

model = load_model("myModel.keras")
model.summary()

# img = plt.imread("LV8/test.png") 

# img_s = img.astype("float32") / 255

# # Ignore alpha channel
# rgb = img_s[:, :, :3]

# # Convert to grayscale using luminance formula
# gray = np.dot(rgb, [0.2989, 0.5870, 0.1140])

# gray = np.expand_dims(gray, -1)

# gray = gray.reshape(1, 28, 28, 1)

# model = load_model("myModel.keras")
# model.summary()

# print(gray)

img = cv2.imread("LV8/test.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (28, 28))

img_resized = img_gray.astype("float32") / 255.0
img_resized = np.expand_dims(img_resized, axis=-1)
img_resized = np.expand_dims(img_resized, axis=0)

predictions = model.predict(img_resized)

print(predictions)

predictions = predictions.argmax(axis=1)

print(predictions)

plt.figure()
plt.imshow(img_resized.reshape(28, 28), cmap='gray')
plt.title("predicted: " + str(predictions[0]) + " actual: 1")
plt.show()