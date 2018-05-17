import codecs
from sklearn import svm

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def main():
    with codecs.open('../msrpc/msr_paraphrase_train.txt', 'r', "utf_8_sig")as file:
        file.readline()
        train_data_first = []
        train_data_second = []
        train_target = list()
        for line in file:
            train_data_first.append(line.split('\t')[3])
            train_data_second.append(line.split('\t')[4].strip())
            train_target.append(int(line.split('\t')[0]))


    with codecs.open('../msrpc/msr_paraphrase_test.txt', 'r', "utf_8_sig")as file:
        file.readline()
        test_data_first = list()
        test_data_second = []
        test_target = list()
        for line in file:
            test_data_first.append(line.split('\t')[3])
            test_data_second.append(line.split('\t')[4].strip())
            test_target.append(int(line.split('\t')[0]))


    # vect = FeatureUnion(
    #     transformer_list=[
    #         ('tfidf', TfidfVectorizer()),
    #         ('stats', PosExtractor()),
    #     ])

    # text_clf = Pipeline([
    #     ('union', vect),
    #     ('clf', LogisticRegression()),
    # ])

    # text_clf.fit(train_data, train_target)
    # predicted = text_clf.predict(test_data)
    # print(np.mean(predicted == test_target))



    # clf = svm.SVC()
    # clf.fit(train_data, train_target)
    # print(clf.predict(test_data))


    s1 = TfidfVectorizer.transform(train_data_first)
    s2 = TfidfVectorizer.transform(train_data_second)
    X = abs(s1 - s2)
    print(X)

if __name__ == '__main__':
    main()
