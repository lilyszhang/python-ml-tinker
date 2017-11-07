import csv
import os
from HTMLParser import HTMLParser
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pyprind
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#strip html tags in a csv and create a new csv of the clean text

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# data = csv.reader(open("storage/featuring/11.1.2017-team-featuring.csv", 'rU'))
# writer = csv.writer(open("storage/featuring/11.1.2017-team-featuring-raw.csv", 'w'))
# fields = data.next()
# toWrite = [['body','opinionated']]
# for row in data:
#     item = row
#     item[0] = strip_tags(item[0])
#     toWrite.append(item)
# writer.writerows(toWrite)

# docs into tokens

def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# csv to dataframe

df = pd.read_csv("movie_data.csv")
X = df.loc[:, 'review'].values
y = df.loc[:, 'sentiment'].values

# df = pd.read_csv("storage/featuring/10.25.2017-team-featuring-raw.csv")
# X_test = df.loc[:, 'body'].values
# y_test = df.loc[:, 'opinionated'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# GridSearchCV and LogisticRegression

stop = stopwords.words('english')
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)], 'vect__stop_words': [stop, None], 'vect__tokenizer': [tokenizer, tokenizer_porter], 'clf__penalty': ['l1','l2'], 'clf__C':[1.0,10.0,100.0]}, {'vect__ngram_range': [(1,1)], 'vect__stop_words': [stop, None], 'vect__tokenizer': [tokenizer, tokenizer_porter], 'vect__use_idf':[False], 'vect__norm': [None], 'clf__penalty': ['l1','l2'], 'clf__C': [1.0,10.0,100.0]}]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy',cv=5,verbose=1,n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print 'Best parameter set: %s' % gs_lr_tfidf.best_params_
print 'CV Accuracy: %.3f' % gs_lr_tfidf.best_score_
clf = gs_lr_tfidf.best_estimator_
print 'Test Accuracy: %.3f' % clf.score(X_test, y_test)

# this can be used to classify without having expected results
predicted = clf.predict(X_test)
print np.mean(predicted == y_test)
