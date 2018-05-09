import codecs
import pandas as pd
from lxml import objectify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

from string import punctuation
from nltk import word_tokenize

import numpy as np

sw = stopwords.words('russian') + list(punctuation)

def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in sw and not w.isdigit()]

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
        class_data = pd.DataFrame({'first_sentence': first_sentence,
                                   'second_sentence': second_sentence,
                                   'similarity': similarity})

        tfidf = TfidfVectorizer(analyzer='word',
                                ngram_range=(1,2),
                                stop_words=sw,
                                lowercase=True,
                                max_features=300)

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
        print(classification_report(y_test, predicted))
        print(predicted.tolist())
        print(y_test.tolist())

if __name__ == '__main__':
    main()