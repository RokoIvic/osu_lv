import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 3.0, 3.0, 1.0], float)
y = np.array([1.0, 2.0, 2.0, 1.0, 1.0], float)

plt.plot(x, y, 'b', linewidth=4.0, linestyle='dashed', color="g", marker='.',                   markerfacecolor='red', markersize=8)

plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Zad1')

plt.show()