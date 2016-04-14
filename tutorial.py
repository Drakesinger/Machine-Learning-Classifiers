# -*- encoding: utf-8 -*-
__author__ = 'horia_000'

from sklearn import datasets

# The categories we want.
categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

twenty_train = datasets.load_files ("C:\\Users\\horia_000\\Documents\\Courses\\LVL3\\3258_IA_et_frameworks\\IA\\TP\\Classification\\text_analytics\\data\\twenty_newsgroups\\20news-bydate-train\\", \
description=None, categories=categories, load_content=True,shuffle=True, encoding='latin-1', decode_error='strict', random_state=42)

print twenty_train.target_names

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(twenty_train.data)
print X_train_counts.shape

print count_vector.vocabulary_.get(u'algorithm')

# Instead of just counting, we will use the frequencies.
from sklearn.feature_extraction.text import TfidfTransformer

term_frequency_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
term_frequency_times_inverse_document_frequency_transformer = TfidfTransformer()

X_train_term_frequency_id = term_frequency_transformer.transform(X_train_counts)
X_train_term_frequency_times_inverse_document_frequency = term_frequency_times_inverse_document_frequency_transformer.fit_transform(X_train_counts)

print "Training data: TFIDF : ",X_train_term_frequency_times_inverse_document_frequency.shape

# Now train a classifier.

from sklearn.naive_bayes import MultinomialNB

clasifier = MultinomialNB().fit(X_train_term_frequency_times_inverse_document_frequency,twenty_train.target)

# Try to predict the outcome on new documents.
docs_new = ['God is love', 'OpenGL is fast']
# We need to extract the features the same way:
X_new_counts = count_vector.transform(docs_new)
X_new_tfidf = term_frequency_times_inverse_document_frequency_transformer.transform(X_new_counts)

predicted = clasifier.predict(X_new_tfidf)

for doc,category in zip(docs_new,predicted):
    print "%r => %s" % (doc,twenty_train.target_names[category])


# Let's avoid all this and just make a pipeline.

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

text_classifier = Pipeline([('vect',CountVectorizer()),
                            ('tfidf',TfidfTransformer()),
                            ('classifier', SGDClassifier(loss="hinge",penalty='l2',alpha=1e-13,n_iter=5,random_state=42))])
text_classifier = text_classifier.fit(twenty_train.data,twenty_train.target)

# Test and show the predicitive accuracy.

import numpy as np
twenty_test = datasets.load_files ("C:\\Users\\horia_000\\Documents\\Courses\\LVL3\\3258_IA_et_frameworks\\IA\\TP\\Classification\\text_analytics\\data\\twenty_newsgroups\\20news-bydate-test\\", \
description=None, categories=categories, load_content=True,shuffle=True, encoding='latin-1', decode_error='strict', random_state=42)

docs_test = twenty_test.data
predicted = text_classifier.predict(docs_test)

print np.mean(predicted == twenty_test.target)

from sklearn import metrics

# Show a report of the metrics.
print metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names)

print twenty_test.target_names
# Show the confusion matrix.
print metrics.confusion_matrix(twenty_test.target, predicted)

