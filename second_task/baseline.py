import codecs, string, nltk, argparse
import pandas as pd
from lxml import objectify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import FeatureUnion

import numpy as np

sw = stopwords.words('russian')

class PosStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = []
        for sentence in posts:
            tokenized = nltk.tokenize.word_tokenize(sentence[0])
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


def main(parser):
    args = parser.parse_args()
    with codecs.open(args.src_train_texts, 'r', args.encoding) as file:
        data = objectify.fromstring(file.read())
        first_sentence = []
        second_sentence = []
        similarity = []

        for doc in data.corpus.paraphrase:
            first_sentence.append(doc.value[3].text)
            second_sentence.append(doc.value[4].text)
            similarity.append(doc.value[6].text)

        first_sentence = tokenize(first_sentence, args.word_type)
        second_sentence = tokenize(second_sentence, args.word_type)

        class_data = pd.DataFrame({'first_sentence': first_sentence,
                                   'second_sentence': second_sentence,
                                   'similarity': similarity})

        if(args.features == "true"):
            feat_union = FeatureUnion(transformer_list=[('tfidf', TfidfVectorizer(analyzer='word',
                                                                                  min_df=3,
                                                                                  ngram_range=(1, args.n),
                                                                                  stop_words=sw)),
                                                        ('cv', CountVectorizer(analyzer='word',
                                                                               ngram_range=(1, args.n))),
                                                        ('pos', PosStats())])
        else:
            feat_union = TfidfVectorizer(analyzer='word',
                                        min_df=3,
                                        ngram_range=(1, args.n),
                                        stop_words=sw)

        BagOfWords = pd.concat([class_data.first_sentence, class_data.second_sentence], axis=0)
        feat_union.fit(BagOfWords)

        train_s1_matrix = feat_union.transform(class_data.first_sentence)
        train_s2_matrix = feat_union.transform(class_data.second_sentence)

        X = abs(train_s1_matrix - train_s2_matrix)
        y = class_data.similarity

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        predicted = lr.predict(X_test)
        with open(args.output, 'a+') as out:
            out.write(str(np.mean(predicted == y_test)) + "\n")
            out.write(classification_report(y_test, predicted))
            out.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-train-texts', action="store", dest="src_train_texts", default='../paraphraser/paraphrases.xml')
    parser.add_argument('--text-encoding', action="store", dest="encoding", default="utf_8")
    parser.add_argument('--word-type', choices=['surface_all', 'surface_no_pm', 'stem'], default="surface_no_pm", action="store", dest="word_type")
    parser.add_argument('-n', type=int, action="store", dest="n", default=2)
    parser.add_argument('--features', choices=['true', 'false'], action="store", default='true')
    parser.add_argument('--laplace', action="store_true")
    parser.add_argument('--unknown-word-freq', type=int, action="store", default=0, dest="word_freq")
    parser.add_argument('-o', action="store", dest="output", default='../second_task/output.txt')

    main(parser)