from Orange.base import Learner
from Orange.data import Table, Domain, Variable, ContinuousVariable, DiscreteVariable, StringVariable
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from tf_core.nltoolkit.helpers import NltkClassifier


class BowDataset:
    def __init__(self,sparse_bow_matrix,labels=None):
        #orange
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
        Y, labels=bow_model.get_document_labels(adc,binary=True)
        domain=Domain([ContinuousVariable.make(a) for a in bow_model.vectorizer.get_feature_names()],
                      DiscreteVariable("class",values=labels))
        new_data = Table.from_numpy(domain, sparse_bow_matrix, Y=Y)

        return new_data #cls(sparse_bow_matrix,labels)

    #def sparce_bow_matrix(self):
    #    return self.sparse_bow_matrix()
    def dense_bow_matrix(self):
        return self.sparse_bow_matrix.toarray()

    def nltk_dataset_with_labels(self):
        train_data=[({},label) for label in self.labels]
        cx=self.sparse_bow_matrix.tocoo() #A sparse matrix in COOrdinate format.
        if cx.shape[0]!=len(self.labels):
            raise Exception("Nekaj gnilega je v dezeli Danski, sporoci maticu.")
        for (i,j,v) in zip(cx.row, cx.col, cx.data):
            train_data[i][0][j]=v   #seting the dict in (list(tuple(dict, str)))
        return train_data

    def nltk_dataset_without_labels(self):
        cx=self.sparse_bow_matrix.tocoo() #A sparse matrix in COOrdinate format.
        train_data=[{} for _ in range(cx.shape[0])]
        for (i,j,v) in zip(cx.row, cx.col, cx.data):
            train_data[i][j]=v   #seting the dict in (list(dict))
        return train_data

    def bow_in_proper_format(self,classifier,no_labels=False):
        #check if classifier can deal with sparse data
        if isinstance(classifier,NltkClassifier):
            return self.nltk_dataset_without_labels() if no_labels else self.nltk_dataset_with_labels()
        elif isinstance(classifier, (GaussianNB, MultinomialNB,  DecisionTreeClassifier)):
            return self.dense_bow_matrix()
        elif isinstance(classifier,Learner):
            # from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table
            # Table(input_dict['file'])
            #
            # def orange_load_dataset_from_arff_string(input_dict):
            #     output_dict = {}
            #     data = arff.loads(input_dict['arff'])
            #     attributes = []
            #     classVar = None
            #     for idx, (att_name, values) in enumerate(data['attributes']):
            #         if values == 'REAL':
            #             att = ContinuousVariable(att_name)
            #         else:
            #             att = DiscreteVariable(att_name, values)
            #         if idx == len(data['attributes']) - 1:
            #             classVar = att
            #         else:
            #             attributes.append(att)
            #     domain = Domain(attributes, classVar)
            #     data = Table.from_list(domain, data['data'])
            #     output_dict['dataset'] = data
            #     return output_dict
            #
            # data = input_dict['data']
            #
            # Y = classifier(data)
            #
            # new_data = Table.from_numpy(data.domain, data.X, Y=Y, metas=data.metas)
            print("hello")
        else: #if latino classifier or a classifier that can deal with sparse data
            return self.sparse_bow_matrix

    def split(self,train_indices,test_indices=None):
        output_train = BowDataset(self.sparse_bow_matrix[train_indices],
                                      self.get_document_labels(train_indices))
        output_test = BowDataset(self.sparse_bow_matrix[test_indices], self.get_document_labels(test_indices)) \
                if test_indices else None
        return output_train,output_test

    def get_document_labels(self,indices):
        return [self.labels[i] for i in indices] if self.labels is not None else []

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
