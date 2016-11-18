import csv
from extract import extract_body
from bayes import NaiveBayes
import os
import logging
import random

logger = logging.getLogger(__name__)

class SpamHamDetector(object):
    def __init__(self, categories, path):
        self.naive_bayes = NaiveBayes(categories)
        self.path = path
        self.classified_examples = dict()


    def train(self):
        with open('{0}/labels.csv'.format(self.path), 'r') as labels_csv:
            reader = csv.DictReader(labels_csv)
            for row in reader:
                label = (row['Prediction'])
                filename = '%s/TR/TRAIN_%s.eml' % (path, row['Id'])
                try:
                    body = extract_body(filename)
                    self.naive_bayes.train(int(label), body)

                except Exception as e:
                    logger.info("Error training email %s: %s", row['Id'], e.message)

    def train_and_evaluate(self):
        all_ids = list(range(1, 2501))
        random.shuffle(all_ids)
        training_ids, labeling_ids = all_ids[:2250], all_ids[2250:]

        with open('{0}/labels.csv'.format(self.path), 'r') as labels_csv:
            reader = csv.DictReader(labels_csv)
            for row in reader:
                label = (row['Prediction'])
                filename = '%s/TR/TRAIN_%s.eml' % (path, row['Id'])
                if int(row['Id']) in training_ids:
                    try:
                        body = extract_body(filename)
                        self.naive_bayes.train(int(label), body)
                    except Exception as e:
                        logger.info("Error training email %s: %s", row['Id'], e.message)

        correct, incorrect = 0, 0
        with open('%s/labels.csv' % self.path, 'r') as labels_csv:
            reader = csv.DictReader(labels_csv)
            for row in reader:
                label = (row['Prediction'])
                filename = '%s/TR/TRAIN_%s.eml' % (path, row['Id'])
                if int(row['Id']) in labeling_ids:
                    try:
                        test_body = extract_body(filename)
                        result = self.naive_bayes.classify(test_body)
                        if result == int(label):
                            correct += 1
                        else:
                            incorrect += 1
                    except Exception as e:
                        logger.info("Error classifying email %s: %s", row['Id'], e.message)
        return self._calculate_results(correct, incorrect)

    def classify(self, size):
        counter = 1
        test = self.path + '/TT/TEST_%s.eml'

        while counter < size+1:
            try:
                test_body = extract_body(test % counter)
                self.classified_examples[str(counter)] = str(self.naive_bayes.classify(test_body))
            except Exception as e:
                logger.info("Error classifying email %s: %s", counter, e.message)
            counter += 1

        self._store_results()

    def display_results(self):
        spam = sum(1 for category in self.classified_examples.values() if category == '0')
        ham = sum(1 for category in self.classified_examples.values() if category == '1')
        return "Spam Emails: %s\nHam Emails: %s\nSpam Percent: %s\nHam Percent: %s" \
               % (spam, ham, (float(spam) / len(self.classified_examples)),
                  (float(ham) / len(self.classified_examples)))

    def _calculate_results(self, correct, incorrect):
        return "correct %s, incorrect %s, performance measurement %s" % (correct,
                                                                         incorrect,
                                                                         (float(correct) / (correct + incorrect)))

    def _store_results(self):
        with open('%s/results.csv' % self.path, 'w+') as resultscsv:
            writer = csv.DictWriter(resultscsv, fieldnames=['id','Prediction'])
            writer.writeheader()
            for example_num, category in self.classified_examples.items():
                writer.writerow({'id': example_num, 'Prediction': category})


if __name__ == "__main__":
    print "starting!"
    path = os.path.dirname(__file__)
    detector = SpamHamDetector([0, 1], path)
    print detector.train_and_evaluate()
    detector.train()
    print "done training!"
    detector.classify(1827)
    print "done classifying!"
    print detector.display_results()
