import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MeCab
from gensim import corpora
from gensim import models
from sklearn.cross_validation import train_test_split

m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

class Preprocessing:
    def __init__(self, path):
        self.path = path
        self.li_folders = os.listdir(path)

        self.num2category_name_dic = {}
        self.category_name2num_dic = {}
        self.documents = []
        self.target = []

        self.columns = ["category", "vec"]
        self.df = pd.DataFrame(columns=self.columns)

    def wakachi(self, text):
        words = []

        node = m.parseToNode(text)

        while node:
            words.append(node.surface)
            node = node.next

        return words

    def make_category_dic(self, category, fldr_name):
        self.num2category_name_dic[category] = fldr_name
        self.category_name2num_dic[fldr_name] = category

    def adjust(self):   # 1
        category = -1

        for dir in self.li_folders:
            category += 1
            self.make_category_dic(category, dir)

            li_files = os.listdir(self.path + dir)

            for f_name in li_files:
                f = open(self.path + dir + "/" + f_name, encoding= "UTF-8")
                text = f.read()

                words = self.wakachi(text)

                self.documents.append(words)
                self.target.append(category)

    def embedding(self):    # 2
        dic = corpora.Dictionary(self.documents)    # 文書毎の単語のリスト(documents)からgensim.corporaを使い単語辞書を作成

        dic.filter_extremes(no_below=20, no_above=0.3)  # 単語辞書から 出現頻度の少ない単語（２０回未満） 及び 出現頻度の多すぎる単語（３０％以上の記事で登場する単語） を排除

        bow_corpus = [dic.doc2bow(d) for d in self.documents]   # bag-of-words のベクトルを作成

        # TF-IDFによる重み付け
        tfidf_model = models.TfidfModel(bow_corpus)
        tfidf_corpus = tfidf_model[bow_corpus]

        # LSIによる次元削減
        lsi_model = models.LsiModel(tfidf_corpus, id2word=dic, num_topics=300)
        lsi_corpus = lsi_model[tfidf_corpus]

        vec = []
        for i in range(len(lsi_corpus)):
            cps_df = pd.DataFrame(lsi_corpus[i], columns=["id", "num"])
            vec.append(list(cps_df["num"].values.flatten()))

        X = np.array(vec)
        y = np.array(self.target)

        return X, y

    def separate(self, X, y):   # 3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)    # トレーニングデータ：テストデータ　＝　７：３

        return X_train, X_test, y_train, y_test