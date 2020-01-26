from __future__ import division
import nltk
import string
import operator
from stempel import StempelStemmer
from functools import reduce
from timeit import default_timer as timer
from enum import Enum


def ilen(iterable):
    return reduce(lambda sum, element: sum + 1, iterable, 0)


def isPunct(word):
    return len(word) == 1 and word in string.punctuation


def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False


class RakeCalculateMethod(Enum):
    Frequency = 1
    Degree = 2
    DegreeByFreq = 3


class RakeKeywordExtractor:

    def __init__(self, word_count, calculate_method, do_lemmatize, stemmer):
        self.stemmer = stemmer
        self.word_count = word_count - 1
        self.calculate_method = calculate_method
        self.do_lemmatize = do_lemmatize
        self.stopwords = set(nltk.corpus.stopwords.words('polish'))
        self.top_fraction = 1  # consider top third candidate keywords by score

    def _generate_candidate_keywords(self, sentences):
        phrase_list = []
        for sentence in sentences:
            if self.do_lemmatize:
                words = map(lambda x: "|" if x in self.stopwords else self.stemmer.stem(x),
                            nltk.word_tokenize(sentence.lower(), 'polish'))
            else:
                words = map(lambda x: "|" if x in self.stopwords else x,
                            nltk.word_tokenize(sentence.lower(), 'polish'))
            phrase = []
            words_list = list(words)
            while any(words_list):
                word = words_list.pop(0)
                if word == "|" or isPunct(word):
                    if len(phrase) > 0:
                        phrase_list.append(phrase)
                        phrase = []
                elif len(phrase) > self.word_count:
                    phrase_list.append(phrase)
                    phrase = []
                    words_list.insert(0, word)
                else:
                    phrase.append(word)
        return phrase_list

    def _calculate_word_scores(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_degree = nltk.FreqDist()
        for phrase in phrase_list:
            degree = ilen(filter(lambda x: not isNumeric(x), phrase)) - 1
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += degree
        for word in word_freq.keys():
            word_degree[word] = word_degree[word] + word_freq[word]  # itself
        # word score = deg(w) / freq(w)
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = {
                RakeCalculateMethod.Frequency: word_freq[word],
                RakeCalculateMethod.Degree: word_degree[word],
                RakeCalculateMethod.DegreeByFreq: word_degree[word] / word_freq[word]
            }[self.calculate_method]

        return word_scores

    def _calculate_phrase_scores(self, phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                phrase_score += word_scores[word]
            phrase_scores[" ".join(phrase)] = phrase_score
        return phrase_scores

    def extract(self, text, incl_scores=False):
        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidate_keywords(sentences)
        word_scores = self._calculate_word_scores(phrase_list)
        phrase_scores = self._calculate_phrase_scores(
            phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(),
                                      key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        if incl_scores:
            return sorted_phrase_scores[0:int(n_phrases / self.top_fraction)]
        else:
            return map(lambda x: x[0],
                       sorted_phrase_scores[0:int(n_phrases / self.top_fraction)])


def test():

    filename = input("Input filename: ")
    word_count = int(input("Podaj maksymalną liczbę słów w fazie kluczowej: "))
    rake_method = RakeCalculateMethod(int(input("Wybierz metodę obliczania wag dla fraz kluczowych : "
                                                "\n1-Częstotliwość\n2-Stopień\n3-stosunek stopnia do częstotliwości\n")))
    support_lemmatization = input("Czy lematyzować słowa?\n") == 'y'
    rake = RakeKeywordExtractor(word_count, rake_method, support_lemmatization, stemmer)
    with open(filename, 'r', encoding='utf-8') as reader:
        contents = reader.read()
        start = timer()
        keywords = rake.extract(contents, incl_scores=True)
        end = timer()

    print("(Keyword, Score)")
    print(*keywords, sep='\n')
    print("Elapsed time in sec: " + str(end - start))


if __name__ == "__main__":
    stemmer = StempelStemmer.polimorf()
    while True:
        test()
