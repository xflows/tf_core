from itertools import izip

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from tf_core.nltoolkit.helpers import NltkClassifier


class BowDataset:
    def __init__(self,sparse_bow_matrix,labels=None):
        self.sparse_bow_matrix=sparse_bow_matrix
        self.labels=labels
    def __len__(self):
        return self.sparse_bow_matrix.shape[0]
    @classmethod
    def from_raw_documents(cls,documents,bow_vectorizer,labels=None):
        sparse_bow_matrix = bow_vectorizer.transform(documents)
        return cls(sparse_bow_matrix,labels)

    @classmethod
    def from_adc(cls,adc,bow_model):
        sparse_bow_matrix = bow_model.vectorizer.transform(bow_model.get_raw_text(adc.documents,join_annotations_with='|##|'))
        labels=bow_model.get_document_labels(adc)

        return cls(sparse_bow_matrix,labels)

    #def sparce_bow_matrix(self):
    #    return self.sparse_bow_matrix()
    def dense_bow_matrix(self):
        return self.sparse_bow_matrix.toarray()

    def nltk_dataset_with_labels(self):
        train_data=[({},label) for label in self.labels]
        cx=self.sparse_bow_matrix.tocoo() #A sparse matrix in COOrdinate format.
        if cx.shape[0]!=len(self.labels):
            raise Exception("Nekaj gnilega je v dezeli Danski, sporoci maticu.")
        for (i,j,v) in izip(cx.row, cx.col, cx.data):
            train_data[i][0][j]=v   #seting the dict in (list(tuple(dict, str)))
        return train_data

    def nltk_dataset_without_labels(self):
        cx=self.sparse_bow_matrix.tocoo() #A sparse matrix in COOrdinate format.
        train_data=[{} for _ in range(cx.shape[0])]
        for (i,j,v) in izip(cx.row, cx.col, cx.data):
            train_data[i][j]=v   #seting the dict in (list(dict))
        return train_data

    def bow_in_proper_format(self,classifier,no_labels=False):
        #check if classifier can deal with sparse data
        if isinstance(classifier,NltkClassifier):
            return self.nltk_dataset_without_labels() if no_labels else self.nltk_dataset_with_labels()
        elif isinstance(classifier, (GaussianNB, MultinomialNB,  DecisionTreeClassifier)):
            return self.dense_bow_matrix()
        else: #if latino classifier or a classifier that can deal with sparse data
            return self.sparse_bow_matrix

    def split(self,train_indices,test_indices=None):
        output_train = BowDataset(self.sparse_bow_matrix[train_indices],
                                      self.get_document_labels(train_indices))
        output_test = BowDataset(self.sparse_bow_matrix[test_indices], self.get_document_labels(test_indices)) \
                if test_indices else None
        return output_train,output_test

    def get_document_labels(self,indices):
        return [self.labels[i] for i in indices] if self.labels!=None else []

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
