# Принимает список списков слов
# Возвращает список списков слов без пунктуации
import re
import string


def remove_punctuation(messages):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    tokenized_docs_no_punctuation = []
    for review in messages:

        new_review = []
        for token in review:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)

        tokenized_docs_no_punctuation.append(new_review)
    return tokenized_docs_no_punctuation


def transform_without_punctuation(corpus):
    corpus_list = []
    for sentence in corpus.values():
        corpus_list.append(sentence.split())

    corpus_list = remove_punctuation(corpus_list)
    return corpus_list


def count_words(corpus_list):
    total_words = 0
    words = set()
    for sentence in corpus_list:
        for word in sentence:
            total_words += 1
            words.add(word)
    return total_words, words
