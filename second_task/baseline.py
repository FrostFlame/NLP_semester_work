import codecs, string, nltk, argparse
import pandas as pd
import scipy
from lxml import objectify
from scipy import sparse
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import pickle

from nltk import word_tokenize

import numpy as np
from sklearn.svm import SVC

sw = stopwords.words('russian')

class PosStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = []
        for sentence in posts:
            tokenized = nltk.tokenize.word_tokenize(sentence)
            tokenized = nltk.pos_tag(tokenized)
            nn, rb, vb, jj = 0, 0, 0, 0
            for word in tokenized:
                if 'NN' in word[1]:
                    nn += 1
                elif 'RB' in word[1]:
                    rb += 1
                elif 'VB' in word[1]:
                    vb += 1
                elif 'JJ' in word[1]:
                    jj += 1
            features.append([nn, rb, vb, jj])
        return features

    def get_feature_names(self):
        return ['nn', 'rb', 'vb', 'jj']


class NgramTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(selfself, posts):
        features = []
        for pair in posts:
            first_sentence, second_sentence = pair.split('-+-')
            first_tokenized = nltk.tokenize.word_tokenize(first_sentence)
            second_tokenized = nltk.tokenize.word_tokenize(second_sentence)
            numerator = 0
            for word in first_tokenized:
                if word in second_tokenized:
                    numerator += 1
            divider = len(first_tokenized) + len(second_tokenized) - numerator
            features.append([numerator/divider, numerator/len(first_tokenized), numerator/len(second_tokenized)])
        return features

    def get_feature_names(self):
        return ['f1', 'f2', 'f3']

def tokenize(list_of_sentences, word_type):
    if(word_type == "surface_all"):
        return list_of_sentences

    stemmer = SnowballStemmer("russian")
    punc = string.punctuation + "«" + "»"
    text = [nltk.wordpunct_tokenize(sentence) for sentence in list_of_sentences]
    text = [[w.lower() for w in sentence if (w not in punc)] for sentence in text]
    new_text = []
    for sentence in text:
        if(word_type == "surface_no_pm"):
            word = [stemmer.stem(i) for i in sentence if (i not in sw and not i.isdigit())]
        else:
            word = [i for i in sentence]
        new_sentence = " ".join(word)
        new_text.append(new_sentence)
    return new_text

def get_data(data, corp):
    first_sentence = []
    second_sentence = []
    similarity = []
    if corp == '../paraphraser/paraphrases.xml':
        for doc in data.corpus.paraphrase:
            first_sentence.append(doc.value[3].text)
            second_sentence.append(doc.value[4].text)
            similarity.append(doc.value[6].text)
    return first_sentence, second_sentence, similarity

def main(parser):
    args = parser.parse_args()
    if args.src_train_texts == '../paraphraser/paraphrases.xml':
        with codecs.open('../paraphraser/paraphrases.xml', 'r', "utf_8") as file:
            data = objectify.fromstring(file.read())

            first_sentence, second_sentence, similarity = get_data(data, args.src_train_texts)
    else:
        with codecs.open('../msrpc/msr_paraphrase_train.txt', 'r', "utf_8_sig")as file:
            file.readline()
            first_sentence = []
            second_sentence = []
            similarity = list()
            for line in file:
                first_sentence.append(line.split('\t')[3])
                second_sentence.append(line.split('\t')[4].strip())
                similarity.append(int(line.split('\t')[0]))

        with codecs.open('../msrpc/msr_paraphrase_test.txt', 'r', "utf_8_sig")as file:
            file.readline()
            for line in file:
                first_sentence.append(line.split('\t')[3])
                second_sentence.append(line.split('\t')[4].strip())
                similarity.append(int(line.split('\t')[0]))

    first_sentence = tokenize(first_sentence, args.word_type)
    second_sentence = tokenize(second_sentence, args.word_type)

    class_data = pd.DataFrame({'first_sentence': first_sentence,
                               'second_sentence': second_sentence,
                               'similarity': similarity})

    if(args.features == "true"):
        feat_union = FeatureUnion(transformer_list=[('tfidf', TfidfVectorizer(analyzer='word',
                                                                              min_df=3,
                                                                              ngram_range=(1, args.n),
                                                                              stop_words=sw,
                                                                              smooth_idf=args.laplace)),
                                                    ('cv', CountVectorizer(analyzer='word',
                                                                           ngram_range=(1, args.n))),
                                                    ('pos', PosStats())])
    else:
        feat_union = TfidfVectorizer(analyzer='word',
                                     min_df=3,
                                     ngram_range=(1, args.n),
                                     stop_words=sw,
                                     smooth_idf=args.laplace)

    BagOfWords = pd.concat([class_data.first_sentence, class_data.second_sentence], axis=0)
    feat_union.fit(BagOfWords)

    train_s1_matrix = feat_union.transform(class_data.first_sentence)
    train_s2_matrix = feat_union.transform(class_data.second_sentence)

    X = abs(train_s1_matrix - train_s2_matrix)
    y = class_data.similarity

    X = X.toarray()
    if (args.features == "true"):
        pairs_of_sentences = []
        for first, second in zip(class_data.first_sentence, class_data.second_sentence):
            pairs_of_sentences.append(first + '-+-' + second)

        double_feat_union = FeatureUnion(transformer_list=[('ngram', NgramTransformer())
                                                           ])
        double_matrix = double_feat_union.transform(pairs_of_sentences)
        # double_matrix = sparse.csr_matrix(double_matrix)
        X = np.hstack((X, double_matrix))

    if args.src_train_texts == '../paraphraser/paraphrases.xml':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
    else:
        X_train = X[:4076]
        X_test = X[4076:]
        y_train = y[:4076]
        y_test = y[4076:]

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    f = open(args.output, 'wb')
    pickle.dump(lr, f)
    f.close()
    predicted = lr.predict(X_test)
    with open('../second_task/output.txt', 'a+') as out:
        out.write(str(np.mean(predicted == y_test)) + "\n")
        out.write(classification_report(y_test, predicted))
        out.write("\n")
    print("~~~~~Task completed~~~~~")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-train-texts', action="store", dest="src_train_texts", default='../paraphraser/paraphrases.xml')
    # parser.add_argument('--src-train-texts', action="store", dest="src_train_texts", default='../msrpc/msrpc_paraphrase_train.txt')
    parser.add_argument('--text-encoding', action="store", dest="encoding", default="utf_8")
    parser.add_argument('--word-type', choices=['surface_all', 'surface_no_pm', 'stem'], default="surface_no_pm", action="store", dest="word_type")
    parser.add_argument('-n', type=int, action="store", dest="n", default=2)
    parser.add_argument('--features', choices=['true', 'false'], action="store", default='true')
    parser.add_argument('--laplace', action="store_true", dest="laplace")
    parser.add_argument('-o', action="store", dest="output", default='../second_task/model.pickle')

    main(parser)