import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X_3 = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X_3[:,0],X_3[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri s 3 grupe')


# 4 grupa
X_4 = generate_data(500, 3)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X_4[:,0],X_4[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri s 4 grupe')


# 2. Podzadatak

# K=3
model_3 = KMeans(n_clusters=3, random_state=0)

model_3.fit(X_3)
X_3_pred = model_3.predict(X_3)
plt.figure()
plt.scatter(X_3[:,0],X_3[:,1], c=X_3_pred)
plt.title('KMeans - K=3')

# K=4
model_4 = KMeans(n_clusters=4, random_state=0)

model_4.fit(X_3)
X_4_pred = model_4.predict(X_3)
plt.figure()
plt.scatter(X_3[:,0],X_3[:,1], c=X_4_pred)
plt.title('KMeans - K=4')

# K=5
model_5 = KMeans(n_clusters=5, random_state=0)

model_5.fit(X_3)
X_5_pred = model_5.predict(X_3)
plt.figure()
plt.scatter(X_3[:,0],X_3[:,1], c=X_5_pred)
plt.title('KMeans - K=5')

# 3. Podzadatak
# Circles
X_2_circles = generate_data(500, 4)
model_2_circles = KMeans(n_clusters=2, random_state=0)

model_2_circles.fit(X_2_circles)
X_2_circles_pred = model_2_circles.predict(X_2_circles)
plt.figure()
plt.scatter(X_2_circles[:,0],X_2_circles[:,1], c=X_2_circles_pred)
plt.title('KMeans - 2 grupe Circles')

# Moons
X_2_moons = generate_data(500, 5)

model_2_moons = KMeans(n_clusters=2, random_state=0)
model_2_moons.fit(X_2_moons)
X_2_moons_pred = model_2_moons.predict(X_2_moons)
plt.figure()
plt.scatter(X_2_moons[:,0],X_2_moons[:,1], c=X_2_moons_pred)
plt.title('KMeans - 2 grupe Moons')


# 4 groups
X_4 = generate_data(500, 3)

model_4 = KMeans(n_clusters=4, random_state=0)
model_4.fit(X_4)
X_4_pred = model_4.predict(X_4)
plt.figure()
plt.scatter(X_4[:,0],X_4[:,1], c=X_4_pred)
plt.title('KMeans - 4 grupe')



plt.show()