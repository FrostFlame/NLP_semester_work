import codecs
import re
import string

from first_task.helper import transform_without_punctuation, count_words


def main():
    with codecs.open('../msrpc/msr_paraphrase_data.txt', 'r', "utf_8_sig")as file:
        file.readline()
        corpus = {}
        for line in file:
            corpus[line.split('\t')[0]] = line.split('\t')[1]
    print('Количество документов', len(corpus))

    corpus_list = transform_without_punctuation(corpus)

    total_words, words = count_words(corpus_list)
    print('Всего слов', total_words)
    print('Уникальных слов', len(words))


    
if __name__ == '__main__':
    main()
