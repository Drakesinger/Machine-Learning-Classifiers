__author__ = 'Horia Mut'

import os
import Tkinter
import tkFileDialog



# defining options for opening a directory
dir_opt = options = {}
options['initialdir'] = os.getcwd()
options['mustexist'] = True
options['title'] = 'Twenty Newsgroups dataset folder'


def get_directory_path():
    Tkinter.Tk().withdraw()  # Close the root window
    in_path = tkFileDialog.askdirectory(**dir_opt)
    return in_path


from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


# Work parallel
def do_parallel(twenty_train, text_classifier):
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3)}

    grid_search_clasifier = GridSearchCV(text_classifier, parameters, n_jobs=-1)
    grid_search_clasifier = grid_search_clasifier.fit(twenty_train.data, twenty_train.target)

    # print twenty_train.target_names[grid_search_clasifier.predict(['God is love'])]

    best_parameters, score, _ = max(grid_search_clasifier.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    # The categories we want.
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

    twenty_train = datasets.load_files(
        get_directory_path() + "/20news-bydate-train", \
        description=None, categories=categories, load_content=True, shuffle=True, encoding='latin-1',
        decode_error='strict', random_state=42)

    text_classifier = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', SGDClassifier(loss='hinge',
                                                      penalty='l2',
                                                      alpha=1e-3,
                                                      n_iter=5,
                                                      random_state=42)), ])

    text_classifier = text_classifier.fit(twenty_train.data, twenty_train.target)

    # do_parallel(twenty_train, text_classifier)

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3), }

    grid_search_clasifier = GridSearchCV(text_classifier, parameters, n_jobs=-1)
    grid_search_clasifier = grid_search_clasifier.fit(twenty_train.data[:400], twenty_train.target[:400])

    print twenty_train.target_names[grid_search_clasifier.predict(['God is love'])]

    # best_parameters, score, _ = max(grid_search_clasifier.grid_scores_, key=lambda x: x[1])
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, best_parameters[param_name]))
