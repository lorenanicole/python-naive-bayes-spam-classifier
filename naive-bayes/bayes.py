from collections import defaultdict
from decimal import Decimal
import nltk
import re


class NaiveBayes(object):
    def __init__(self, categories):
        self.words = defaultdict(dict)
        self.categories = self._create_categories(categories)
        self.training_examples = 0
        self.unique_words = set()

    def _create_categories(self, categories):
        categories = {category: {'total': 0, 'word_count': 0}
                      for category in categories}
        return categories

    def train(self, category, text):
        text = self._tokenize_text(text)  # TODO: stem words

        self._increment_unique_word_count(text)  # Laplace Smoothing
        self._increment_word_frequency(category, text)
        self._increment_category_count(category)
        self._increment_category_word_count(category, len(text))

        self.training_examples += 1

    def _tokenize_text(self, text):
        text = re.findall(r"[\w']+", text)
        words = []
        for word in text:
            if word and word not in nltk.corpus.stopwords.words('english'):
                words.append(word)
        return words

    def _increment_word_frequency(self, category, words):
        for word in words:
            if self.words[word].get(category):
                self.words[word][category] += 1
            else:
                self.words[word][category] = 1

    def _increment_unique_word_count(self, text):
        self.unique_words = set(list(self.unique_words) + text)

    def _increment_category_count(self, category):
        self.categories[category]['total'] += 1

    def _increment_category_word_count(self, category, number):
        if self.categories[category].get('word_count'):
            self.categories[category]['word_count'] += number
        else:
            self.categories[category]['word_count'] = number

    def classify(self, text):
        text = self._tokenize_text(text)

        probabilities = {}
        for cat, cat_data in self.categories.iteritems():
            category_prob = self._get_category_probability(cat_data['total'])
            predictors_likelihood = self._get_predictors_probability(cat, text)
            probabilities[cat] = category_prob * predictors_likelihood

        return 1 if probabilities[1] > probabilities[0] else 0

    def _get_category_probability(self, count):
        # Can make use of logarithm in lieu of Python's decimal object to avoid
        # Floating point underflow
        # e.g. return log(class_prior_prob)
        return Decimal(float(count)) / Decimal(self.training_examples + len(self.categories.keys()))

    def _get_predictors_probability(self, category, text):
        word_count = self.categories[category]['word_count'] + len(self.unique_words)
        likelihood = 1
        for word in text:
            if not self.words.get(word) or not self.words[word].get(category):
                smoothed_freq = 1  # Laplace smoothing
            else:
                smoothed_freq = 1 + self.words[word][category]
            likelihood *= Decimal(float(smoothed_freq)) / Decimal(word_count)
            # floating point underflow!! EEE!
            # http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
            # likelihood *= Decimal(float(self.words[word][category])) / Decimal(word_count)
            # print category, log(predictor_likelihood)
        return likelihood