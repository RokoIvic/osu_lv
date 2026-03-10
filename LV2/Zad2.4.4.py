import numpy as np
import matplotlib.pyplot as plt

blackSquare = np.zeros([50, 50])
whiteSquare = np.ones([50, 50])

finalImage = np.hstack((np.vstack((blackSquare, whiteSquare)), np.vstack((whiteSquare,  blackSquare))))

plt.imshow(finalImage, cmap="gray")
plt.show()