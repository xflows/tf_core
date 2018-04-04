from django.conf import settings

import nltk
from tagging_common import universal_word_tagger_hub
from nltk.corpus import wordnet
from pattern.vector import stem, PORTER, LEMMA
from pattern.en import parse
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer

#from tagging_common_parallel import universal_word_tagger_hub
from tf_core.nltoolkit.helpers import NltkRegexpStemmer


def stem_lemma_tagger_hub(input_dict):
    if input_dict['tagger'].__class__.__name__=="LatinoObject": #check if this is a latino object
        from tf_latino.latino.library_gen import latino_tag_adc_stem_lemma
        from workflows.tasks import executeFunction

        return latino_tag_adc_stem_lemma(input_dict) if not settings.USE_WINDOWS_QUEUE \
            else executeFunction.apply_async([latino_tag_adc_stem_lemma,input_dict],queue="windows").wait()
    else:
        adc = input_dict['adc']
        tagger_dict = input_dict['tagger']
        input_annotation = input_dict['element_annotation']
        pos_annotation = input_dict.get('pos_annotation')
        output_annotation = input_dict['output_feature']
        return universal_word_tagger_hub(adc,tagger_dict,input_annotation,output_annotation,pos_annotation)

# STEMMERS
def nltk_lancaster_stemmer(input_dict):
    """
    A word stemmer based on the Lancaster stemming algorithm.
    """
    return {'tagger':
                {'object': nltk.stem.lancaster.LancasterStemmer(),
                 'function': 'stem', }
    }


def nltk_porter_stemmer(input_dict):
    """
    A word stemmer based on the Porter stemming algorithm.
    """
    return {'tagger':
                {'object': nltk.stem.porter.PorterStemmer(),
                 'function': 'stem',
                }
    }


def nltk_isri_stemmer(input_dict):
    """
    ISRI Arabic stemmer based on algorithm: Arabic Stemming without a root dictionary.
    Information Science Research Institute. University of Nevada, Las Vegas, USA.
    """
    return {'tagger':
                {'object': nltk.stem.isri.ISRIStemmer(),
                 'function': 'stem',
                }
    }


def nltk_regexp_stemmer(input_dict):
    """
    A stemmer that uses regular expressions to identify morphological
    affixes.  Any substrings that match the regular expressions will
    be removed.

    :type regexp: str or regexp
    :param regexp: The regular expression that should be used to
        identify morphological affixes.
    :type min: int
    :param min: The minimum length of string to stem
    """

    regexp = input_dict["regexp"] #default ing$|s$|e$|able$
    min = int(input_dict["min"]) #default 4
    return {'tagger':
                {'object': NltkRegexpStemmer(regexp, min=min),
                 'function': 'stem',
                }
    }


def nltk_rslp_stemmer(input_dict):
    """
   A stemmer for Portuguese.

   """
    return {'tagger':
                {'object': nltk.stem.rslp.RSLPStemmer(),
                 'function': 'stem',
                }}


def nltk_snowball_stemmer(input_dict):
    """
    Snowball Stemmer
    The following languages are supported:
    Danish, Dutch, English, Finnish, French, German,
    Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian,
    Spanish and Swedish.

    The algorithm for English is documented here:
    Porter, M. \"An algorithm for suffix stripping.\"
    Program 14.3 (1980): 130-137.

    The algorithms have been developed by Martin Porter.
    These stemmers are called Snowball, because Porter created
    a programming language with this name for creating
    new stemming algorithms. There is more information available
    at http://snowball.tartarus.org/

    :param language: The language whose subclass is instantiated.
    :type language: str or unicode
    :param ignore_stopwords: If set to True, stopwords are
                         not stemmed and returned unchanged.
                         Set to False by default.
    :type ignore_stopwords: bool
    """

    language = input_dict["language"]
    ignore_stopwords = input_dict["ignore_stopwords"] == "true"

    if language == "danish":
        stemmer = nltk.stem.snowball.DanishStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "dutch":
        stemmer = nltk.stem.snowball.DutchStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "english":
        stemmer = nltk.stem.snowball.EnglishStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "finnish":
        stemmer = nltk.stem.snowball.FinnishStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "french":
        stemmer = nltk.stem.snowball.FrenchStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "german":
        stemmer = nltk.stem.snowball.GermanStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "hungarian":
        stemmer = nltk.stem.snowball.HungarianStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "italian":
        stemmer = nltk.stem.snowball.ItalianStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "norwegian":
        stemmer = nltk.stem.snowball.NorwegianStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "portuguese":
        stemmer = nltk.stem.snowball.PortugueseStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "romanian":
        stemmer = nltk.stem.snowball.RomanianStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "russian":
        stemmer = nltk.stem.snowball.RussianStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "spanish":
        stemmer = nltk.stem.snowball.SpanishStemmer(ignore_stopwords=ignore_stopwords)
    elif language == "swedish":
        stemmer = nltk.stem.snowball.SwedishStemmer(ignore_stopwords=ignore_stopwords)

    return {'tagger':
            {'object': stemmer,
             'function': 'stem',
            }}


class WordnetLemmatizer:
    def __init__(self, pos_annotation):
        self.pos_annotation = pos_annotation
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        self.morphy_tag = {'NN':wordnet.NOUN, 'NNS':wordnet.NOUN,
                  'NNP':wordnet.NOUN, 'NNPS':wordnet.NOUN, 'JJ':wordnet.ADJ,
                  'JJR':wordnet.ADJ, 'JJS':wordnet.ADJ, 'VB':wordnet.VERB,
                  'VBD':wordnet.VERB, 'VBG':wordnet.VERB, 'VBN':wordnet.VERB,
                  'VBP':wordnet.VERB, 'VBZ':wordnet.VERB,'RB':wordnet.ADV,
                  'RBR':wordnet.ADV, 'RBS':wordnet.ADV}
    
    def lemmatize(self, lemma, **kwargs):
        if kwargs and self.pos_annotation:
            pos_tag = kwargs[self.pos_annotation]
            if pos_tag in self.morphy_tag:
                return self.lemmatizer.lemmatize(lemma, self.morphy_tag[pos_tag])
        return self.lemmatizer.lemmatize(lemma)


def nltk_wordnet_lemmatizer(input_dict):
    """
    WordNet Lemmatizer
    Lemmatize using WordNet's built-in morphy function.
    Returns the input word unchanged if it cannot be found in WordNet.
    """

    pos_annotation = input_dict.get('pos_annotation')
    return {'tagger':
                {'object': WordnetLemmatizer(pos_annotation),
                 'function': 'lemmatize',
                }}

class LemmagenLemmatizer:
   
    def lemmatize(self, word):
        lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_ENGLISH)
        return lemmatizer.lemmatize(word)



def lemmagen_lemmatizer(input_dict):
    return {'tagger':
                {'object': LemmagenLemmatizer(),
                 'function': 'lemmatize',
                }}    


class PatternLemmatizer:
    def lemmatize(self, word):
        if word == '/':
            return word
        if len(word) == 0:
            return word
        word = word.replace('/', '')
        return parse(word, lemmata=True).split('/')[4]


def pattern_lemmatizer(input_dict):
    """
    WordNet Lemmatizer
    Lemmatize using WordNet's built-in morphy function.
    Returns the input word unchanged if it cannot be found in WordNet.
    """
    return {'tagger':
                {'object': PatternLemmatizer(),
                 'function': 'lemmatize',
                }}


class PatternPorterStemmer:
    def stem(self, word):
        return stem(word, stemmer = PORTER)


def pattern_porter_stemmer(input_dict):
    """
    WordNet Lemmatizer
    Lemmatize using WordNet's built-in morphy function.
    Returns the input word unchanged if it cannot be found in WordNet.
    """
    return {'tagger':
                {'object': PatternPorterStemmer(),
                 'function': 'stem',
                }}


class DefaultLemmatizer:
    #Baseline lemmatizer, doesn't do anything, returns unchanged word
    def lemmatize(self, word):
        return word


def default_lemmatizer(input_dict):
    return {'tagger':
                {'object': DefaultLemmatizer(),
                 'function': 'lemmatize',
                }}































