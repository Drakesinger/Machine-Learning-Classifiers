# -*- encoding: utf-8 -*-

import os
import Tkinter
import tkFileDialog

# We only have two kinds of text: positive text and negative text
from preprocessing import categories

from preprocessing import start_preprocessing as preprocess_data

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

import numpy as np
from sklearn import metrics

__author__ = 'Horia Mut'


def get_dataset_path(type_of_data=' '):
    '''
    Starts a GUI dialog asking for the folder path where the datasets are located.
    :param type_of_data: String defining the type of data used: training or test
    :return: path to dataset folder.
    '''
    # Defining options for opening a directory.
    dir_opt = options = {}
    options['initialdir'] = os.getcwd()
    options['mustexist'] = True
    options['title'] = 'Processed ' + type_of_data + 'dataset folder:'

    Tkinter.Tk().withdraw()  # Close the root window
    dataset_path = tkFileDialog.askdirectory(**dir_opt)
    return dataset_path


# Paths. Hardcoded as not important for now.
# dataset_path = get_dataset_path()
dataset_path_train = os.getcwd() + "/Processed/train"  # Faster than writing it every time
dataset_path_test = os.getcwd() + "/Processed/test"

# Encoding.
encoding = 'latin-1'


# Work parallel
def do_parallel(dataset_training, text_classifier):
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'classifier__alpha': (1e-10, 1e-13)}

    grid_search = GridSearchCV(text_classifier, parameters, n_jobs=-1)
    grid_search = grid_search.fit(dataset_training.data, dataset_training.target)

    # print "Grid Scores:"
    # print grid_search.grid_scores_
    # print_best_scores(grid_search, parameters)

    return grid_search


def print_best_scores(grid_search, parameters):
    print "Best grid parameters:\nparameter name : value"
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == "__main__":
    preprocess_data();

    # Load the training dataset.
    print "Loading training dataset."
    sentiment_train = datasets.load_files(dataset_path_train, \
                                          description=None, categories=categories, load_content=True, shuffle=True,
                                          encoding=encoding, decode_error='strict', random_state=42)
    print "Training dataset loaded."
    # print sentiment_train.target_names
    #
    # #################################
    # # Vector the data
    # #################################
    # count_vector = CountVectorizer()
    # X_train_counts = count_vector.fit_transform(sentiment_train.data)
    # print X_train_counts.shape
    #
    # print count_vector.vocabulary_.get(u'nom')
    #
    # # Instead of just counting, we will use the frequencies.
    #
    # term_frequency_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    # term_frequency_times_inverse_document_frequency_transformer = TfidfTransformer()
    #
    # X_train_term_frequency_id = term_frequency_transformer.transform(X_train_counts)
    # X_train_term_frequency_times_inverse_document_frequency = term_frequency_times_inverse_document_frequency_transformer.fit_transform(
    #     X_train_counts)
    #
    # print "Training data: TFIDF : ", X_train_term_frequency_times_inverse_document_frequency.shape
    # #################################
    # # Now train a classifier.
    # #################################
    #
    # clasifier = MultinomialNB().fit(X_train_term_frequency_times_inverse_document_frequency, sentiment_train.target)
    #
    # # Try to predict the outcome on new documents.
    # docs_new = ['La chose est mal', "Wow c'est super bien"]
    # # We need to extract the features the same way:
    # X_new_counts = count_vector.transform(docs_new)
    # X_new_tfidf = term_frequency_times_inverse_document_frequency_transformer.transform(X_new_counts)
    #
    # predicted = clasifier.predict(X_new_tfidf)
    #
    # for doc, category in zip(docs_new, predicted):
    #     print "%r => %s" % (doc, sentiment_train.target_names[category])
    #

    ##################################################################
    # Let's avoid all this and just make a pipeline.
    ##################################################################

    text_classifier = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('classifier',
                                 # LinearSVC(C=1000)
                                 # MultinomialNB()
                                 SGDClassifier(loss="hinge", penalty='l2', alpha=1e-13, n_iter=5, random_state=42)
                                 )])
    # Fit the classifier to the training data and target (pos, neg).
    text_classifier = text_classifier.fit(sentiment_train.data, sentiment_train.target)

    # Start our parallel grid search.
    grid_search = do_parallel(sentiment_train, text_classifier)

    # Test and show the predictive accuracy.
    print "Loading test dataset."
    sentiment_test = datasets.load_files(dataset_path_test, \
                                         description=None, categories=categories, load_content=True, shuffle=True,
                                         encoding=encoding, decode_error='strict', random_state=42)
    print "Dataset has been loaded."
    docs_test = sentiment_test.data
    # Start prediction.
    print "Starting prediction analysis."
    predicted = grid_search.predict(docs_test)

    # Show the mean accuracy.
    print "Mean accuracy: ",np.mean(predicted == sentiment_test.target)

    # Print the classification report.
    print(metrics.classification_report(sentiment_test.target, predicted,
                                        target_names=sentiment_test.target_names))

    # Print and plot the confusion matrix.
    cm = metrics.confusion_matrix(sentiment_test.target, predicted)
    print cm

    # Plot the confusion matrix. Not worth it here.
    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.show()
