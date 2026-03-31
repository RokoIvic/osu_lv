import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)
plt.figure(1)

plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', edgecolor='k', s=20, cmap='coolwarm', c=y_train, alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', s=20, cmap='viridis', c=y_test, alpha=0.7)
plt.xlabel("x1")
plt.ylabel("x2")

# b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

y_test_p = LogRegression_model.predict( X_test )

# c)
print("Koeficijenti:", LogRegression_model.coef_)
print("Intercept:", LogRegression_model.intercept_)

plt.figure(2)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', edgecolor='k', s=20, cmap='coolwarm', c=y_train, alpha=0.7)

theta0 = LogRegression_model.intercept_[0]
theta1, theta2 = LogRegression_model.coef_[0]

# d)
x1_vals = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 100)
x2_vals = -(theta0 + theta1 * x1_vals) / theta2

plt.plot(x1_vals, x2_vals, 'k-', label='Granica odluke')

cm = confusion_matrix(y_test , y_test_p)
disp = ConfusionMatrixDisplay (confusion_matrix=cm)
disp.plot()

print("Tocnost: ", accuracy_score(y_test , y_test_p))
print(classification_report(y_test , y_test_p))

# e)
colors = np.array([])
for i in range(len(y_test)):
    if (y_test[i] == y_test_p[i]):
        colors = np.append(colors, 'green')
    else:
        colors = np.append(colors, 'black')

plt.figure(4)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', s=20, c=colors, alpha=0.7)


plt.show()