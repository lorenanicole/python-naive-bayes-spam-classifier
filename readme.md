###Basic Naive Bayes Classifier in Python

Project is written with Python 2.7.

This approach makes use of pre-labeled data provided by the [Kaggle Classroom spam detection challenge](https://inclass.kaggle.com/c/adcg-ss14-challenge-02-spam-mails-detection/data).

For setup create a virtualenv with the requirements:

```
virtualenv nbenv
source nbenv/bin/activate
pip install -r pathway/to/naive-baves/requirements.txt
```

To run the Naive Bayes classifier: 

```
cd naive-bayes
python spam_detector.py
```
###Python 3 Jupyter Notebook

The Python 2.7 project has been ported to Python 3 and can be run in the Jupyter notebook.

First you will want to create a Python3 virtualenv:

```
pyenv-3.5 python3env  # Update 3.5 with your version of Python 3
source python3env/bin/activate  # Name your env whatever you like!
pip3 install -r requirements.txt 
``` 
Then start the notebook!

```
jupyter notebook
```

### Notes on Python Naive Bayes Implementation

You can have the detector either train and evaluate itself against the training data (using 90% of the pre-labeled data as training and 10% to label) with: 

```
detector.train_and_evaluate()
```

Or you can train against the entire labeled data set (2500 emails) and classify on the unlabeled data (1827 emails).

```
detector.train()
detector.classify(1827)  # Number of emails to classify
```

Ham has a label of 1 while Spam has a label of 0.

###How Naive Bayes Implemented

This solution makes use of [Python's 2.7 Decimal module](https://docs.python.org/2/library/decimal.html), which is used for floating point arithmetic. (Prevents [floating point underflow](http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)!)

Inside the NaiveBayes#train method each document has common stop words removed using [NLTK](install http://www.nltk.org/install.html). The words have not yet been [stemmed](http://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python) as this is a forthcoming feature.

Only the corpus of words are used as selectors to determine if an email is spam or ham. 

To prevent words with 0 frequency from miscontruing the results, Laplace smoothing is applied to increment each 0 frequency word to 1.

