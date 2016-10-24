import copy
from itertools import izip
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import nltk
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB





def simulate_cf_pickling(obj_to_pickle,compress_object=False):
    from base64 import b64encode, b64decode
    from zlib import compress, decompress
    from cPickle import dumps,loads

    if not compress_object:
        return loads(b64decode(b64encode(dumps(obj_to_pickle))))
    else:
        return loads(decompress(b64decode(b64encode(compress(dumps(obj_to_pickle))))))

#python manage.py export_package workflows/nltoolkit/db/package_data.json nltoolkit
#python manage.py export_package workflows/literature_based_discovery/db/package_data.json literature_based_discovery
#python manage.py celery worker -l info



if __name__=="__main__":
    """quick test, pickling and depickling"""
    from cPickle import dumps,loads
    from base64 import b64encode, b64decode

    dc=DocumentCorpus([Document("name","text",[Annotation(1,2,'token',{'stopword': True})],{})],{'created_at':'now'})
    dc2=loads(b64decode(b64encode(dumps(dc))))
