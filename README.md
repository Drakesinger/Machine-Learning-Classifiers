# Machine-Learning Classifiers
Text data machine learning with python for the IA course.

## Goal
Use scikit learn to train a classifier in order to detect wether a text epresses positive or negative feelings.

### Assets
Every machine learning system requires data to work with. This data has been provided and can be found in the Assets folder.
In our case the data is made up of annotated text files containing:
- 1 word from the original text per line.
- The type of word it is: Determinant, Adverb, Verb, Noun etc.
- The infinitive form of the word: lower case, verb in infinitive form.

So each row within the file is of the following syntax: `[word]\t[type]\t[infinitive form]\n`

### What was done
Steps: 

1. **Assets processing.**
2. Extract features from the data.
3. Vectorize these features.
4. **Build model.**
5. Train model with training data.
6. Test model with test data.

### Details
In order to correctly determine if the text expresses a negative or positive feeling one needs to establish a *Part of Speech* (PoS) which says what type of words from a language we want to consider. There is a high chance that Nouns for example do not contribute much to the emotion that the author of a certain text wants to pass along. Adjectives however could be much more important. Negation is also tricky since the text *"Not bad, not bad at all."* would show a positive feeling rather than a negative one, even if we have 2 negations and 2 negative adjectives in the sentence. So, annotating the text correctly, choosing the "good" PoS will make the prediction model more accurate.

During the 1st step **Assets processing.** we decide which PoS we take and we eliminate all words that do not belong.
The text files are read, the words not belonging to the PoS are deleted and only the infinitive form of all words are taken. Ponctuation is ignored as well. Once this is done, the words are written randomly, separated by a `\n` character within a new file of the same name at the following path: `[project_folder]\Assets\Processed\[neg,pos]\[filename]`.

Steps 2,3 and 4 are made in one shot using the scikit-learn **pipeline** where we specifiy the **feature extractors** (*tokenizer* and the *transformer*) and **classifier** (the *model*) we want to use.
In our case, we extract the number of occurences of each word, we count the frequency of those occcurences times the inverse of the ammount of times they appear within the corpus (all the training texts used). This means that words that appear frequently are less important that words that appear rarely. Take "and" for example, it doesn't really express and emotion does it?

The **classifier** used is the SVM ([Support Vector Machine](http://scikit-learn.org/stable/modules/svm.html#svm-mathematical-formulation))  with the SGD model ([Stochastic Gradient Descent](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html), [read more about it here](http://scikit-learn.org/stable/modules/sgd.html#sgd)) with the parameters used in the tutorial. Parameter changing can improve the accuracy of the classifier.

Once the classifier has been fit with the training data, we start it in parallel performing a grid search for the best parameters to use with the classifier. Results are shown after the train process has finished.

Once this is done, the testing [corpus](https://en.wikipedia.org/wiki/Text_corpus) is loaded and used with the classifier.
Accuracy of the classifier and the confusion matrix is shown after.
The [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) tells us how many of texts classified as negative are estimated to be positive and vice versa by the classifier.




