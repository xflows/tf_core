#!/usr/bin/env python

# from nltk.classify.megam import config_megam, call_megam
#from nltk.classify.weka import WekaClassifier, config_weka
from datetime import time
from nltk import ELEProbDist
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.positivenaivebayes import PositiveNaiveBayesClassifier
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.classify.rte_classify import rte_classifier, rte_features, RTEFeatureExtractor
from nltk.classify.util import accuracy, apply_features, log_likelihood
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.maxent import (MaxentClassifier, BinaryMaxentFeatureEncoding,
                                  TypedMaxentFeatureEncoding,
                                  ConditionalExponentialClassifier)
#################
#OBSERVATIONS
#################
#GaussianNB is not a good fit for document classification at all, since tf-idf values are non-negative frequencies;
# use MultinomialNB instead, and maybe try BernoulliNB. scikit-learn comes with a document classification example that,
# incidentally, uses tf-idf weighting using the built-in TfidfTransformer.
#################
from tf_core.nltoolkit.helpers import NltkClassifier, DictionaryProbDist


def nltk_naive_bayes_classifier(input_dict):
    """
    A classifier based on the Naive Bayes algorithm.  In order to find the
    probability for a label, this algorithm first uses the Bayes rule to
    express P(label|features) in terms of P(label) and P(features|label):

    |                       P(label) * P(features|label)
    |  P(label|features) = ------------------------------
    |                              P(features)

    The algorithm then makes the 'naive' assumption that all features are
    independent, given the label:

    |                       P(label) * P(f1|label) * ... * P(fn|label)
    |  P(label|features) = --------------------------------------------
    |                                         P(features)

    Rather than computing P(featues) explicitly, the algorithm just
    calculates the denominator for each label, and normalizes them so they
    sum to one:

    |                       P(label) * P(f1|label) * ... * P(fn|label)
    |  P(label|features) = --------------------------------------------
    |                        SUM[l]( P(l) * P(f1|l) * ... * P(fn|l) )


    """
    estimator = ELEProbDist #TODO estimator
    classifier=NltkClassifier(NaiveBayesClassifier,estimator=estimator)

    return {'classifier': classifier}


from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier, megam
#from nltk_trainer import basestring
#from nltk_trainer.classification.multi import AvgProbClassifier

#
# classifier_choices = ['NaiveBayes', 'DecisionTree', 'Maxent'] + MaxentClassifier.ALGORITHMS
#
# dense_classifiers = set(['ExtraTreesClassifier', 'GradientBoostingClassifier',
#                          'RandomForestClassifier', 'GaussianNB', 'DecisionTreeClassifier'])
# verbose_classifiers = set(['RandomForestClassifier', 'SVC'])
#
# try:
#     import svmlight  # do this first since svm module makes ugly errors
#     from nltk.classify.svm import SvmClassifier
#
#     classifier_choices.append('Svm')
# except:
#     pass
#
# try:
#     from nltk.classify import scikitlearn
#     from sklearn.feature_extraction.text import TfidfTransformer
#     from sklearn.pipeline import Pipeline
#     from sklearn import ensemble, feature_selection, linear_model, naive_bayes, neighbors, svm, tree
#
#     classifiers = [
#         ensemble.ExtraTreesClassifier,
#         ensemble.GradientBoostingClassifier,
#         ensemble.RandomForestClassifier,
#         linear_model.LogisticRegression,
#         #linear_model.SGDClassifier, # NOTE: this seems terrible, but could just be the options
#         naive_bayes.BernoulliNB,
#         naive_bayes.GaussianNB,
#         naive_bayes.MultinomialNB,
#         neighbors.KNeighborsClassifier,  # TODO: options for nearest neighbors
#         svm.LinearSVC,
#         svm.NuSVC,
#         svm.SVC,
#         tree.DecisionTreeClassifier,
#     ]
#     sklearn_classifiers = {}
#
#     for classifier in classifiers:
#         sklearn_classifiers[classifier.__name__] = classifier
#
#     classifier_choices.extend(sorted(['sklearn.%s' % c.__name__ for c in classifiers]))
# except ImportError as exc:
#     sklearn_classifiers = {}

def train_classifier(input_dict):
    classifier=input_dict['classifier']
    training_bow_dataset = input_dict['training_data'] #BowDataset
    training_data=training_bow_dataset.bow_in_proper_format(classifier)

    if isinstance(classifier,NltkClassifier):
        trained_classifier=classifier.train(training_data)
        return {'trained_classifier': trained_classifier}
    elif input_dict['classifier'].__module__.startswith('sklearn'):
        classifier.fit(training_data, training_bow_dataset.labels)
        return {'trained_classifier': classifier}
    else:
        from tf_latino.latino.library_gen import latino_train_classifier
        return latino_train_classifier(input_dict)

def convert_to_probdists(csf,y_probas):
    classes = csf.classes_
    return [DictionaryProbDist(dict((classes[i], p)
                                   for i, p in enumerate(y_proba))) for y_proba in y_probas]

#apply_classifier already exists in orange package
def apply_bow_classifier(input_dict):
    trained_classifier = input_dict['trained_classifier']
    if input_dict.get('probability','true')=='true':
        try:
            trained_classifier.set_params(probability=True)
        except (ValueError,AttributeError):   #some classifiers don't have probability parameter
            pass

    testing_bow_dataset = input_dict['testing_dataset']
    testing_dataset=testing_bow_dataset.bow_in_proper_format(trained_classifier,no_labels=True)

    classifier_package=input_dict['trained_classifier'].__module__
    if trained_classifier.__class__.__name__=='LatinoObject':  #check if this is a latino object
        from tf_latino.latino.library_gen import latino_predict_classification
        return latino_predict_classification(input_dict)
    elif classifier_package.startswith("sklearn"):
        #a=trained_classifier.predict(testing_dataset)
        #example: http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html
        try:
            results = [DictionaryProbDist.from_probabilities_and_classes(example_predictions,trained_classifier.classes_)
                       for example_predictions in trained_classifier.predict_proba(testing_dataset)]
        except AttributeError:
            results = [DictionaryProbDist.from_prediction_and_classes(example_prediction,trained_classifier.classes_)
                       for example_prediction in trained_classifier.predict(testing_dataset)]
        #results=convert_to_probdists(trained_classifier,_results)
    elif isinstance(trained_classifier,NltkClassifier):
        results=[DictionaryProbDist(prob_dict=dpd._prob_dict,normalize=True)
                 for dpd in trained_classifier.prob_classify_many(testing_dataset)]
    else:
        raise Exception("What are you connecting me to then?")

    return {'labeled_dataset': None, 'predictions': results}



def extract_classifier_name(input_dict):
    import re

    in1=input_dict['classifier']
    clsf=in1.__class__.__module__+'.'+in1.__class__.__name__ if not in1.__class__.__name__=='LatinoObject' else in1.name


    out2=re.search(r'[A-Za-z\.0-9]+',clsf).group()

    spl=out2.split('.')

    if spl[0]=='sklearn':
      spl[0]='scikit-learn'
    elif spl[0]=='Latino':
      spl[0]='LATINO'
    if spl[-1]=='naive':
      spl[-1]='Gaussian Naive Bayes Classifier'
    if spl[-1]=='LinearSVC':
      spl[-1]='SVM Linear Classifier'

    return {'classifier_name': '['+spl[0]+'] '+spl[-1]}


def extract_actual_and_predicted_values(input_dict):
    actual=input_dict['dataset']
    predicted=input_dict['predictions'] #[a.max() for a in input_dict['predictions']]
    return {'actual_and_predicted': [list(actual.Y), list(predicted.Y)]}