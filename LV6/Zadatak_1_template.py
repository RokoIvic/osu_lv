import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("LV6/Social_Network_Ads.csv")
print(data.info())

data.hist()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("LOG_REG Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()

# Zad6.5.1
# 1)
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)

print("KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu KNN
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()

# 2)
# K = 1
KNN_model_1K = KNeighborsClassifier(n_neighbors=1)
KNN_model_1K.fit(X_train_n, y_train)

y_train_p = KNN_model_1K.predict(X_train_n)
y_test_p = KNN_model_1K.predict(X_test_n)

plot_decision_regions(X_train_n, y_train, classifier=KNN_model_1K)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=1) Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()

# K = 100
KNN_model_100K = KNeighborsClassifier(n_neighbors=100)
KNN_model_100K.fit(X_train_n, y_train)

y_train_p = KNN_model_100K.predict(X_train_n)
y_test_p = KNN_model_100K.predict(X_test_n)

plot_decision_regions(X_train_n, y_train, classifier=KNN_model_100K)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=100) Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()

# Zad6.5.2
pipe_KNN = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

param_grid_KNN = {
    'model__n_neighbors': [1, 5, 10, 50, 100]
}

grid = GridSearchCV(pipe_KNN, param_grid_KNN, cv=5, scoring='accuracy')

grid.fit(X_train, y_train)

print("Best parameters KNN:", grid.best_params_)
print("Best CV score KNN:", grid.best_score_)
print("Test accuracy KNN:", grid.score(X_test, y_test))

# Zad6.5.3
SVM_model_1 = svm.SVC(kernel='rbf', gamma = 1, C=1.0)
SVM_model_1.fit(X_train_n, y_train)

y_train_p = SVM_model_1.predict(X_train_n)
y_test_p = SVM_model_1.predict(X_test_n)

plot_decision_regions(X_train_n, y_train, classifier=SVM_model_1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM_1 Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


SVM_model_2 = svm.SVC(kernel='sigmoid', gamma = 3, C=0.1)
SVM_model_2.fit(X_train_n, y_train)

y_train_p = SVM_model_2.predict(X_train_n)
y_test_p = SVM_model_2.predict(X_test_n)

plot_decision_regions(X_train_n, y_train, classifier=SVM_model_2)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM_2 Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()

# Zad6.5.4
pipe_SVM = Pipeline([
    ('scaler', StandardScaler()),
    ('model', svm.SVC())
])

param_grid_SVM = {
    'model__C': [0.1, 0.5, 1, 5, 10],
    'model__gamma': [1, 0.5, 0.1, 0.05, 0.01]
}

grid = GridSearchCV(pipe_SVM, param_grid_SVM, cv=5, scoring='accuracy')

grid.fit(X_train, y_train)

print("Best parameters SVM:", grid.best_params_)
print("Best CV score SVM:", grid.best_score_)
print("Test accuracy SVM:", grid.score(X_test, y_test))



plt.show()