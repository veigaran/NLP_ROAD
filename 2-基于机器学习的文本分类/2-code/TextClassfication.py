#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2022-03-15 17:26
@Author:Veigar
@File: test.py
@Github:https://github.com/veigaran
"""

# 载入接下来分析用的库
import pandas as pd
import numpy as np
import jieba
import time
import xgboost as xgb
# from tqdm import tqdm
from sklearn.svm import SVC
# from keras.models import Sequential
# from keras.layers.recurrent import LSTM, GRU
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.embeddings import Embedding
# from keras.layers.normalization import BatchNormalization
# from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from sklearn.externals import joblib
import joblib


# from sklearn.naive_bayes import MultinomialNB

class TextClassifier:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.stwlist = [line.strip() for line in open('stopword.txt',
                                                      'r', encoding='utf-8').readlines()]
        self.xtrain, self.xvalid, self.ytrain, self.yvalid = None, None, None, None
        self.xtrain_tfv, self.xvalid_tfv, self.xtrain_ctv, self.xvalid_ctv = None, None, None, None
        # self.model_path = model_path

    def pre_process(self):
        # data = pd.read_excel('./corpus/data.xlsx', 'sheet1')
        df = pd.read_csv('2.csv')
        data = df.iloc[:20000]
        print(data)
        jieba.enable_parallel(64)  # 并行分词开启
        data['文本分词'] = data['正文'].apply(lambda i: jieba.cut(i))
        data['文本分词'] = [' '.join(i) for i in data['文本分词']]
        print(data.head())
        df = data[['分类', '文本分词']]
        df.to_csv('fenci.csv')

    def get_data(self):
        data = pd.read_csv('fenci.csv').iloc[:10000]
        lbl_enc = preprocessing.LabelEncoder()
        y = lbl_enc.fit_transform(data.分类.values)
        self.xtrain, self.xvalid, self.ytrain, self.yvalid = train_test_split(data.文本分词.values, y,
                                                                              stratify=y,
                                                                              random_state=42,
                                                                              test_size=0.1, shuffle=True)

    def tf_idf(self):
        tfv = TfidfVectorizer(min_df=3,
                              max_df=0.5,
                              max_features=None,
                              ngram_range=(1, 2),
                              use_idf=True,
                              smooth_idf=True,
                              stop_words=self.stwlist)
        # 使用TF-IDF来fit训练集和测试集
        tfv.fit(list(self.xtrain) + list(self.xvalid))
        self.xtrain_tfv = tfv.transform(self.xtrain)
        self.xvalid_tfv = tfv.transform(self.xvalid)
        self.save_model(tfv, './model/tfidf.m')

    def word_count(self):
        ctv = CountVectorizer(min_df=3,
                              max_df=0.5,
                              ngram_range=(1, 2),
                              stop_words=self.stwlist)
        # 使用Count Vectorizer来fit训练集和测试集（半监督学习）
        ctv.fit(list(self.xtrain) + list(self.xvalid))
        self.xtrain_ctv = ctv.transform(self.xtrain)
        self.xvalid_ctv = ctv.transform(self.xvalid)

    def linear_regression(self):
        # 利用提取的TFIDF特征来fit一个简单的Logistic Regression
        clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
        clf.fit(self.xtrain_tfv, self.ytrain)
        predictions = clf.predict_proba(self.xvalid_tfv)
        self.evaluate(self.yvalid, clf.predict(self.xvalid_tfv), predictions)
        self.save_model(clf, './model/lr.m')

    def svm(self):
        # 使用SVD进行降维，components设为120，对于SVM来说，SVD的components的合适调整区间一般为120~200
        svd = decomposition.TruncatedSVD(n_components=120)
        svd.fit(self.xtrain_tfv)
        xtrain_svd = svd.transform(self.xtrain_tfv)
        xvalid_svd = svd.transform(self.xvalid_tfv)
        # 对从SVD获得的数据进行缩放
        scl = preprocessing.StandardScaler()
        scl.fit(xtrain_svd)
        xtrain_svd_scl = scl.transform(xtrain_svd)
        xvalid_svd_scl = scl.transform(xvalid_svd)
        # 调用下SVM模型
        clf = SVC(C=1.0, probability=True)  # since we need probabilities
        clf.fit(xtrain_svd_scl, self.ytrain)
        predictions = clf.predict_proba(xvalid_svd_scl)
        self.evaluate(self.yvalid, clf.predict(self.xvalid_tfv), predictions)

    def xgboost(self):
        # 基于tf-idf特征，使用xgboost
        clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                                subsample=0.8, nthread=10, learning_rate=0.1)
        clf.fit(self.xtrain_tfv.tocsc(), self.ytrain)
        predictions = clf.predict_proba(self.xvalid_tfv.tocsc())
        self.evaluate(self.yvalid, clf.predict(self.xvalid_tfv), predictions)

    def evaluate(self, yvalid, y_predict, predictions):
        # print(classification_report(yvalid, y_predict,
        #                             target_names=['news_story', 'news_culture', 'news_entertainment', 'news_sports',
        #                                           'news_finance',
        #                                           'news_house', 'news_car', 'news_edu', 'news_tech', 'news_military',
        #                                           'news_travel', 'news_world', 'news_agriculture',
        #                                           'news_game']))
        print(classification_report(yvalid, y_predict))
        print("logloss: %0.3f " % self.multiclass_logloss(yvalid, predictions))

    @staticmethod
    def multiclass_logloss(actual, predicted, eps=1e-15):
        """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
        :param actual: 包含actual target classes的数组
        :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
        """
        # Convert 'actual' to a binary array if it's not already:
        if len(actual.shape) == 1:
            actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
            for i, val in enumerate(actual):
                actual2[i, val] = 1
            actual = actual2

        clip = np.clip(predicted, eps, 1 - eps)
        rows = actual.shape[0]
        vsota = np.sum(actual * np.log(clip))
        return -1.0 / rows * vsota

    def grid_search(self):
        mll_scorer = metrics.make_scorer(self.multiclass_logloss, greater_is_better=False, needs_proba=True)
        # SVD初始化
        svd = TruncatedSVD()
        # Standard Scaler初始化
        scl = preprocessing.StandardScaler()
        # 再一次使用Logistic Regression
        lr_model = LogisticRegression()
        # 创建pipeline
        clf = pipeline.Pipeline([('svd', svd),
                                 ('scl', scl),
                                 ('lr', lr_model)])

        param_grid = {'svd__n_components': [120, 180],
                      'lr__C': [0.1, 1.0, 10],
                      'lr__penalty': ['l1', 'l2']}
        # 网格搜索模型（Grid Search Model）初始化
        model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                             verbose=10, n_jobs=-1, refit=True, cv=2)

        # fit网格搜索模型
        model.fit(self.xtrain_tfv, self.ytrain)  # 为了减少计算量，这里我们仅使用xtrain
        print("Best score: %0.3f" % model.best_score_)
        print("Best parameters set:")
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def save_model(self, model, path):
        joblib.dump(model, path)
        print("Model is done")

    def model_predict(self, text, classfication_model_path, tf_path):
        """
        :param text: 单个文本
        :param model: 分类模型
        :param tf: 向量器
        :return: 返回预测概率和预测类别
        """
        tfidf_model = joblib.load(tf_path)
        classfication_model = joblib.load(classfication_model_path)
        # tfidf = tfidf_model.transform(sents).toarray()
        text1 = [" ".join(jieba.cut(text))]
        # 进行tfidf特征抽取
        text2 = tfidf_model.transform(text1)
        predict_type = classfication_model.predict(text2)[0]
        print(predict_type)
        # return predict_type


if __name__ == '__main__':
    start = time.time()
    Model = TextClassifier('')
    Model.get_data()
    Model.tf_idf()
    # Model.grid_search()
    Model.linear_regression()
    # Model.model_predict('交警查车也有“潜规则”，他们就爱拦这些车，几乎一查一个准', './model/lr.m', './model/tfidf.m')
    end = time.time()
    print('程序执行时间: ', end - start)

'''

def number_normalizer(tokens):
    """ 将所有数字标记映射为一个占位符（Placeholder）。
    对于许多实际应用场景来说，以数字开头的tokens不是很有用，
    但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))





'''
