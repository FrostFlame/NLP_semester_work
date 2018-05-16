import codecs, string, nltk
import pandas as pd
from lxml import objectify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from nltk import word_tokenize

import numpy as np

sw = stopwords.words('russian')

def tokenize(list_of_sentences):
    stemmer = SnowballStemmer("russian")
    punc = string.punctuation + "«" + "»"
    text = [nltk.wordpunct_tokenize(sentence) for sentence in list_of_sentences]
    text = [[w.lower() for w in sentence if (w not in punc)] for sentence in text]
    new_text = []
    for sentence in text:
        word = [stemmer.stem(i) for i in sentence if (i not in sw and not i.isdigit())]
        new_sentence = " ".join(word)
        new_text.append(new_sentence)
    return new_text

def main():
    with codecs.open('../paraphraser/paraphrases.xml', 'r', "utf_8") as file:
        data = objectify.fromstring(file.read())
        first_sentence = []
        second_sentence = []
        similarity = []

        for doc in data.corpus.paraphrase:
            first_sentence.append(doc.value[3].text)
            second_sentence.append(doc.value[4].text)
            similarity.append(doc.value[6].text)

        first_sentence = tokenize(first_sentence)
        second_sentence = tokenize(second_sentence)

        class_data = pd.DataFrame({'first_sentence': first_sentence,
                                   'second_sentence': second_sentence,
                                   'similarity': similarity})

        tfidf = TfidfVectorizer(analyzer='word',
                                min_df=3,
                                ngram_range=(1, 2),
                                stop_words=sw)

        BagOfWords = pd.concat([class_data.first_sentence, class_data.second_sentence], axis=0)
        tfidf.fit(BagOfWords)

        train_s1_tfidf = tfidf.transform(class_data.first_sentence)
        train_s2_tfidf = tfidf.transform(class_data.second_sentence)

        X = abs(train_s1_tfidf - train_s2_tfidf)
        y = class_data.similarity

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        predicted = lr.predict(X_test)
        print(np.mean(predicted == y_test))

        vect = CountVectorizer()
        vect.fit(BagOfWords)

        train_s1_tfidf = vect.transform(class_data.first_sentence)
        train_s2_tfidf = vect.transform(class_data.second_sentence)

        X = abs(train_s1_tfidf - train_s2_tfidf)
        y = class_data.similarity

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        predicted = lr.predict(X_test)
        print(np.mean(predicted == y_test))

        print(classification_report(y_test, predicted))
        print(predicted.tolist())
        print(y_test.tolist())

        # X = []
        # for i in range(len(first_sentence)):
        #     string = " ".join([first_sentence[i], second_sentence[i]])
        #     X.append(string)
        #
        # y = similarity
        #
        # feat_union = FeatureUnion(
        #     transformer_list=[
        #         ('tfidf', tfidf),
        #         ('vect', CountVectorizer(analyzer='word',
        #                                  ngram_range=(1, 2)))
        #     ])
        #
        # text_clf = Pipeline([
        #     ('union', feat_union),
        #     ('clf', LogisticRegression()),
        # ])
        #
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
        #
        # text_clf.fit(X_train, y_train)
        # predicted = text_clf.predict(X_test)
        #
        # print(np.mean(predicted == y_test))
        # print(classification_report(y_test, predicted))
        # print(predicted.tolist())
        # print(y_test)
if __name__ == '__main__':
    main()