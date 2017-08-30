from preprocessing import Preprocessing
from sklearn.neighbors import KNeighborsClassifier

path = '../data/res/text/'

pp = Preprocessing(path)

pp.adjust()

X, y = pp.embedding()

X_train, X_test, y_train, y_test = pp.separate(X, y)

print(len(X_train),len(X_test))


knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")

knn.fit(X_train, y_train)

train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)

print("[ KNN ]\ntrain score: {0}\ntest score: {1}".format(train_score, test_score))