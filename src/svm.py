from preprocessing import Preprocessing
from sklearn.svm import SVC

path = '../data/res/text/'

pp = Preprocessing(path)

pp.adjust()

X, y = pp.embedding()

X_train, X_test, y_train, y_test = pp.separate(X, y)

print(len(X_train),len(X_test))

# ------------------------
#   線形SVM
# ------------------------

svm = SVC(kernel='linear', C = 1.0, random_state=0)     # 線形SVMのインスタンス生成

svm.fit(X_train, y_train)      # 学習

train_score = svm.score(X_train, y_train)       # 評価（学習データ）
test_score = svm.score(X_test, y_test)      # 評価（テストデータ）

print("[ 線形SVM ]\ntrain score: {0}\ntest score: {1}".format(train_score, test_score))

# ------------------------
#   カーネルSVM
# ------------------------

k_svm = SVC(kernel='rbf', C = 10.0, random_state=0, gamma = 0.10)     # RBFカーネルによるSVMのインスタンス生成

k_svm.fit(X_train, y_train)      # 学習

train_score = k_svm.score(X_train, y_train)       # 評価（学習データ）
test_score = k_svm.score(X_test, y_test)      # 評価（テストデータ）

print("[ カーネルSVM ]\ntrain score: {0}\ntest score: {1}".format(train_score, test_score))