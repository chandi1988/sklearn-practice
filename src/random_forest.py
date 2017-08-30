from preprocessing import Preprocessing
from sklearn.ensemble import RandomForestClassifier

path = '../data/res/text/'

pp = Preprocessing(path)

pp.adjust()

X, y = pp.embedding()

X_train, X_test, y_train, y_test = pp.separate(X, y)

print(len(X_train),len(X_test))


forest = RandomForestClassifier(criterion="entropy", n_estimators=10, random_state=1, n_jobs=2)

forest.fit(X_train, y_train)

train_score = forest.score(X_train, y_train)
test_score = forest.score(X_test, y_test)

print("[ ランダムフォレスト ]\ntrain score: {0}\ntest score: {1}".format(train_score, test_score))