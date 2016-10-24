import copy
import json

from document import Document


class DocumentCorpus:
    def __init__(self, documents,features):
        self.documents=documents
        self.features=features

    def __unicode__(self):
        #for i in itemDir:
        return 'Documents; {0}' % (self.documents)

    def split(self,train_indices,test_indices=None):
        output_train = DocumentCorpus(copy.deepcopy([self.documents[i] for i in train_indices]),
                                      copy.deepcopy(self.features))
        output_test = DocumentCorpus(copy.deepcopy([self.documents[i] for i in test_indices]),
                                      copy.deepcopy(self.features)) if test_indices else None

        return output_train,output_test

    def get_labels(self):
        return json.loads(self.features['Labels']) if 'Labels' in self.features else []

    def get_document_labels(self):
        return [doc.get_first_label() for doc in self.documents]

    def __getstate__(self):
        print "get state!!"
        minimized_docs=[d.__minimize__() for d in self.documents]
        return json.dumps([minimized_docs,self.features])

    def __setstate__(self,value):
        print "set state!!"
        minimized_docs,self.features=json.loads(value)
        self.documents=[Document.__from_minimized__(d) for d in minimized_docs]


