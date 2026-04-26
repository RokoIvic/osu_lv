import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from keras.models import load_model


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

model = load_model("myModel.keras")
model.summary()

predictions = model.predict(x_test_s)

predictions = predictions.argmax(axis=1)

print(predictions)

for i in range(500):
    if(predictions[i] != y_test[i]):
        plt.figure()
        plt.imshow(x_test_s[i].reshape(28, 28), cmap='gray')
        plt.title("predicted: " + str(predictions[i]) + " actual: " + str(y_test[i]))
        

plt.show()