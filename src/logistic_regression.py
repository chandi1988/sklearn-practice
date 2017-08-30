from preprocessing import Preprocessing
from sklearn.linear_model import LogisticRegression

path = '../data/res/text/'

pp = Preprocessing(path)

pp.adjust()

X, y = pp.embedding()

X_train, X_test, y_train, y_test = pp.separate(X, y)

print(len(X_train),len(X_test))


lr = LogisticRegression(C = 1000, random_state = 0)

lr.fit(X_train, y_train)    #学習

train_score = lr.score(X_train,y_train)     #評価（学習データ）
test_score = lr.score(X_test,y_test)     #評価（テストデータ）

print("train score: {0}\ntest score: {1}".format(train_score, test_score))