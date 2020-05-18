from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=6)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

con_matrix = confusion_matrix(y_test, y_pred)
print(con_matrix)

report = classification_report(y_test, y_pred)
print(report)
