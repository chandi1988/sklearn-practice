from preprocessing import Preprocessing

path = '../data/res/text/'

pp = Preprocessing(path)

pp.adjust()

X, y = pp.embedding()

X_train, X_test, y_train, y_test = pp.separate(X, y)

print(len(X_train),len(X_test))