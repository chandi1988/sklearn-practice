from preprocessing import Preprocessing
from sklearn.tree import DecisionTreeClassifier

path = '../data/res/text/'

pp = Preprocessing(path)

pp.adjust()

X, y = pp.embedding()

X_train, X_test, y_train, y_test = pp.separate(X, y)

print(len(X_train),len(X_test))


tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)

tree.fit(X_train, y_train)

train_score = tree.score(X_train, y_train)
test_score = tree.score(X_test, y_test)

print("[ 決定木 ]\ntrain score: {0}\ntest score: {1}".format(train_score, test_score))