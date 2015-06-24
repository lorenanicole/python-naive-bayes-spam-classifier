import csv
from extract import extract_body
from bayes import NaiveBayes
import os
import logging

logger = logging.getLogger(__name__)

class SpamHamDetector(object):
    def __init__(self, categories, path):
        self.naive_bayes = NaiveBayes(categories)
        self.path = path
        self.classified_examples = dict()


    def train(self):
        with open('%s/labels.csv' % self.path, 'r') as labels_csv:
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
        correct, incorrect = 0, 0
        with open('%s/labels.csv' % self.path, 'r') as labels_csv:
            reader = csv.DictReader(labels_csv)
            for row in reader:
                label = (row['Prediction'])
                filename = '%s/TR/TRAIN_%s.eml' % (path, row['Id'])
                if int(label) < 2251:
                    try:
                        body = extract_body(filename)
                        self.naive_bayes.train(int(label), body)
                    except Exception as e:
                        logger.info("Error training email %s: %s", row['Id'], e.message)
                if int(row['Id']) > 2250:
                    try:
                        test_body = extract_body(filename)
                        result = self.naive_bayes.classify(test_body)
                        if result == int(row['Prediction']):
                            correct += 1
                        else:
                            incorrect += 1
                    except Exception as e:
                        logger.info("Error classifying email %s: %s", row['Id'], e.message)
                # print row['Id']
        return "correct %s\n, incorrect %s, performance measurement %s" % (correct,
                                                                           incorrect,
                                                                           (float(correct) / (correct + incorrect)))

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

    def _store_results(self):
        with open('%s/results.csv' % self.path, 'w+') as resultscsv:
            writer = csv.DictWriter(resultscsv, fieldnames=['id','Prediction'])
            writer.writeheader()
            for example_num, category in self.classified_examples.items():
                writer.writerow({'id': example_num, 'Prediction': category})

    def display_results(self):
        spam = sum(1 for category in self.classified_examples.values() if category == '0')
        ham = sum(1 for category in self.classified_examples.values() if category == '1')
        return "Spam Emails: %s\n Ham Emails: %s \nSpam Percent: %s\nHam Percent: %s" \
               % (spam, ham, (float(spam) / len(self.classified_examples)),
                  (float(ham) / len(self.classified_examples)))



if __name__ == "__main__":
    print "starting!"
    path = os.path.dirname(__file__)
    detector = SpamHamDetector([0,1], path)
    # print detector.train_and_evaluate()
    # correct 249, incorrect 1, performance measurement 0.996
    detector.train()
    print "done training!"
    detector.classify(1827)
    print "done classifying!"
    print detector.display_results()
    # "Spam Emails: %s\n Ham Emails: %s \nSpam Percent: %s\nHam Percent: %s"