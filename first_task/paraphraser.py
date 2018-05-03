import codecs

from lxml import objectify

from first_task.helper import transform_without_punctuation, count_words


def main():
    with codecs.open('../paraphraser/corpus.xml', 'r', "utf_8_sig") as file:
        data = objectify.fromstring(file.read())
        corpus = {}
        for doc in data.corpus.sentence:
            corpus[int(doc.value[0].text)] = doc.value[1].text
        print('Количество документов', len(corpus))

        corpus_list = transform_without_punctuation(corpus)

        total_words, words = count_words(corpus_list)
        print('Всего слов', total_words)
        print('Уникальных слов', len(words))


if __name__ == '__main__':
    main()