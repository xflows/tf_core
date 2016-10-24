import re

import nltk


class TokenSplitter(object):
    '''Used for splitting tokens in BoW construction.'''
    regex_parser=None

    def __init__(self):
        self.regex_parser=re.compile("^[_A-Za-z0-9']+$")
    def __call__(self, doc):
        if doc.startswith("Substance P"):
            sdfasdfa=333
        return [d for d in doc.split('|##|') if self.regex_parser.match(d)]

class NltkCorpus():
    """ Wrapper for Nltk corpora. In Nltk 3.x Nltk corpora is not picklable.
    """
    corpus_name=""
    corpus=None
    _corpus_methods=[]
    def __init__(self,name):
        self.corpus_name=name
        self._corpus_methods=dir(getattr(nltk.corpus,self.corpus_name))

    def _corpus(self):
        self.corpus = self.corpus or getattr(nltk.corpus,self.corpus_name)
        return self.corpus

    def __getattr__(self, name):
        if not name in self._corpus_methods:
            raise AttributeError
        else:
            def method():
                return getattr(self._corpus(),name)()
            return method

    def __repr__(self):
        return "NltkCorpus wrapper for "+self.corpus_name+" dataset"

    def __getstate__(self):
        return {'corpus_name': self.corpus_name,'_corpus_methods': self._corpus_methods}


class NltkRegexpTokenizer():
    """ Wrapper for Nltk RegexTokenizer. Python's regular expressions are not picklable.
    """
    _pattern=''
    _kargs={}

    def __init__(self,pattern,**kargs):
        self._pattern=pattern
        self._kargs=kargs

    def span_tokenize(self,text):
        return nltk.RegexpTokenizer(self._pattern,**self._kargs).span_tokenize(text)

class NltkRegexpStemmer():
    """ Wrapper for Nltk RegexStemmer. Python's regular expressions are not picklable.
    """
    _pattern=''
    _kargs={}

    def __init__(self,pattern,**kargs):
        self._pattern=pattern
        self._kargs=kargs

    def stem(self,text):
        return nltk.stem.RegexpStemmer(self._pattern,**self._kargs).stem(text)


class NltkClassifier():
    """ This is a wrapper for Nltk classifiers. Nltk classifiers do not have an appropriate __init__
        method which could save kargs and use them latter when .train(train_data) is called.
    """
    _classifier=None
    _kargs={}

    def __init__(self,csf,**kargs):
        self._classifier=csf
        self._kargs=kargs

    def train(self,training_data):
        self._classifier=self._classifier.train(training_data,**self._kargs)
        return self

    def prob_classify_many(self,testing_dataset):
        return self._classifier.prob_classify_many(testing_dataset)

    def __repr__(self):
        return '<NltkClassifier object for %s>' % (self._classifier)

from nltk.probability import DictionaryProbDist

class DictionaryProbDist(DictionaryProbDist):
    def __repr__(self):
        return '<ProbDist: %s>' % str(self._prob_dict)

    @classmethod
    def from_prediction_and_classes(cls,prediction,classes):
        prob_dict={}
        for i,klass in enumerate(classes):
            prob_dict[unicode(klass)]=1. if klass==prediction else 0.
        return cls(prob_dict=prob_dict)

    @classmethod
    def from_probabilities_and_classes(cls,predictions,classes):
        prob_dict={}
        for i,klass in enumerate(classes):
            prob_dict[unicode(klass)]=predictions[i]
        return cls(prob_dict=prob_dict)




from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize import TextTilingTokenizer
from nltk.tokenize import TreebankWordTokenizer


class SpanTokenizeMixin():

    def span_tokenize(self, s):
        r"""
        Return the offsets of the tokens in *s*, as a sequence of ``(start, end)``
        tuples, by splitting the string at each successive match of *regexp*.

            #>>> from workflows.textflows import StanfordTokenizer
            #>>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
            #... two of them.\n\nThanks.'''
            #>>> list(StanfordTokenizer().span_tokenize(s))
            #[(0, 4), (5, 12), (13, 17), (18, 23), (24, 26), (27, 30), (31, 36),
            #(38, 44), (45, 48), (49, 51), (52, 55), (56, 58), (59, 64), (66, 73)]

        :param s: the string to be tokenized
        :type s: str
        :rtype: iter(tuple(int, int))
        """
        tokens = self.tokenize(s)
        right = 0

        for token in tokens:
            offset=s[right:].find(token)
            if offset<0:
                sadfa=45
            else:
                #raise Exception('Negative offset!')
                left = right + offset #s[right:].find(token)
                right = left + len(token)
                yield left, right


class StanfordTokenizer(SpanTokenizeMixin, StanfordTokenizer):
    pass


class TextTilingTokenizer(SpanTokenizeMixin, TextTilingTokenizer):
    pass


class TreebankWordTokenizer(SpanTokenizeMixin, TreebankWordTokenizer):
    pass


