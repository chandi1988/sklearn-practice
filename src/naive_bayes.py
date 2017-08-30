from preprocessing import Preprocessing
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

path = '../data/res/text/'

pp = Preprocessing(path)

pp.adjust()

X, y = pp.embedding()

X_train, X_test, y_train, y_test = pp.separate(X, y)

print(len(X_train),len(X_test))



# ---------------------------
#   正規分布（ガウシアン分布）
# ---------------------------

g = GaussianNB()
g.fit(X_train, y_train)

train_score = g.score(X_train, y_train)
test_score = g.score(X_test, y_test)

print("[ 正規分布 ]\ntrain score: {0}\ntest score: {1}".format(train_score, test_score))

# ---------------------------
#   ベルヌーイ分布
# ---------------------------

b = BernoulliNB()
b.fit(X_train, y_train)

train_score = b.score(X_train, y_train)
test_score = b.score(X_test, y_test)

print("[ ベルヌーイ分布 ]\ntrain score: {0}\ntest score: {1}".format(train_score, test_score))

# ---------------------------
#   多項分布
# ---------------------------

# m = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
# m.fit(X_train, y_train)
#
# train_score = m.score(X_train, y_train)
# test_score = m.score(X_test, y_test)
#
# print("[ 多項分布 ]\ntrain score: {0}\ntest score: {1}".format(train_score, test_score))