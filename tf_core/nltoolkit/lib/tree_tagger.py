# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the TreeTagger POS-tagger
#
# Copyright (C) Mirko Otto
# Author: Mirko Otto <dropsy@gmail.com>

"""
A Python module for interfacing with the Treetagger by Helmut Schmid.
"""

import os
import subprocess
from subprocess import Popen, PIPE
from django.conf import settings
import re

from nltk.internals import find_binary, find_file
from nltk.tag.api import TaggerI
from collections import defaultdict

def tUoB(obj, encoding='utf-8'):
    if isinstance(obj, basestring):
        if not isinstance(obj, unicode):
            obj = unicode(obj, encoding)
    return obj

_treetagger_url = 'http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/'

_treetagger_languages = {
u'latin-1':['latin', 'latinIT', 'mongolian', 'swahili'],
u'utf-8' : ['bulgarian', 'dutch', 'english', 'estonian', 'finnish', 'french', 'galician', 'german', 'italian', 'polish', 'russian', 'slovak', 'slovak2', 'spanish']}

"""The default encoding used by TreeTagger: utf-8. u'' means latin-1; ISO-8859-1"""
_treetagger_charset = [u'utf-8', u'latin-1']

interpelations = {'NP':'NNP', 'NPS':'NNPS', 'PP':'PRP', 'PP$':'PRP$', 
                  'VD':'VB', 'VDD':'VBD', 'VDG':'VBG', 'VDN':'VBN', 'VDZ': 'VBZ', 'VDP':'VBP',
                  'VH':'VB', 'VHD':'VBD', 'VHG':'VBG', 'VHN':'VBN', 'VHZ': 'VBZ', 'VHP':'VBP',
                  'VV':'VB', 'VVD':'VBD', 'VVG':'VBG', 'VVN':'VBN', 'VVZ': 'VBZ', 'VVP':'VBP' }
                   

class TreeTagger(TaggerI):
    ur"""
    A class for pos tagging with TreeTagger. The input is the paths to:
     - a language trained on training data
     - (optionally) the path to the TreeTagger binary
     - (optionally) the encoding of the training data (default: utf-8)

    This class communicates with the TreeTagger binary via pipes.

    Example:

    .. doctest::
        :options: +SKIP

        >>> from treetagger import TreeTagger
        >>> tt = TreeTagger(encoding='utf-8',language='english')
        >>> tt.tag(u'What is the airspeed of an unladen swallow ?')
        [[u'What', u'WP', u'What'],
         [u'is', u'VBZ', u'be'],
         [u'the', u'DT', u'the'],
         [u'airspeed', u'NN', u'airspeed'],
         [u'of', u'IN', u'of'],
         [u'an', u'DT', u'an'],
         [u'unladen', u'JJ', u'<unknown>'],
         [u'swallow', u'NN', u'swallow'],
         [u'?', u'SENT', u'?']]

    .. doctest::
        :options: +SKIP

        >>> from treetagger import TreeTagger
        >>> tt = TreeTagger()
        >>> tt.tag(u'Das Haus ist sehr schön und groß. Es hat auch einen hübschen Garten.')
        [[u'Das', u'ART', u'd'],
         [u'Haus', u'NN', u'Haus'],
         [u'ist', u'VAFIN', u'sein'],
         [u'sehr', u'ADV', u'sehr'],
         [u'sch\xf6n', u'ADJD', u'sch\xf6n'],
         [u'und', u'KON', u'und'],
         [u'gro\xdf', u'ADJD', u'gro\xdf'],
         [u'.', u'$.', u'.'],
         [u'Es', u'PPER', u'es'],
         [u'hat', u'VAFIN', u'haben'],
         [u'auch', u'ADV', u'auch'],
         [u'einen', u'ART', u'ein'],
         [u'h\xfcbschen', u'ADJA', u'h\xfcbsch'],
         [u'Garten', u'NN', u'Garten'],
         [u'.', u'$.', u'.']]
    """

    def __init__(self, path_to_home=None, language='english', 
                 encoding='utf-8', verbose=False, abbreviation_list=None, widget_id=None, params="C:\TreeTagger\\lib", trained=False):
        """
        Initialize the TreeTagger.

        :param path_to_home: The TreeTagger binary.
        :param language: Default language is german.
        :param encoding: The encoding used by the model. Unicode tokens
            passed to the tag() and batch_tag() methods are converted to
            this charset when they are sent to TreeTagger.
            The default is utf-8.

            This parameter is ignored for str tokens, which are sent as-is.
            The caller must ensure that tokens are encoded in the right charset.
        """
        treetagger_paths = ['C:\TreeTagger\\bin', '.', '/usr/bin', '/usr/local/bin', '/opt/local/bin',
                        '/Applications/bin', '~/bin', '~/Applications/bin',
                        '~/work/TreeTagger/cmd', '~/tree-tagger/cmd']
        treetagger_paths = map(os.path.expanduser, treetagger_paths)
        self._abbr_list = abbreviation_list
        self.widget_id = widget_id
        self.params = params
        self.trained = trained

        try:
            self._encoding = encoding
            if language=='english':
                treetagger_bin_name = 'tag-english.bat'
                train_treetagger_bin_name = 'train-tree-tagger.exe'

        except KeyError as e:
                raise LookupError('NLTK was unable to find the TreeTagger bin!')

        self._treetagger_bin = find_binary(
            treetagger_bin_name, path_to_home,
            env_vars=('TREETAGGER', 'TREETAGGER_HOME'),
            searchpath=treetagger_paths,
            url=_treetagger_url,
            verbose=verbose)

        self._train_treetagger_bin = find_binary(
            train_treetagger_bin_name, path_to_home,
            env_vars=('TREETAGGER', 'TREETAGGER_HOME'),
            searchpath=treetagger_paths,
            url=_treetagger_url,
            verbose=verbose)


    def train(self, sentences):
        lexicon = defaultdict(list)
        train_path = os.path.join(settings.TREE_TAGGER, str(self.widget_id) + ".train")
        if os.path.exists(train_path):
            os.remove(train_path)
        f = open(train_path, 'a')
        for sent in sentences:
            for i, token in enumerate(sent):
                word, tag = token
                if i == len(sent) - 1 and tag == '.':
                    tag = 'SENT'
                f.write(word + '\t' + tag + '\n')
                if tag not in lexicon[word]:
                    lexicon[word].append(tag)
        f.close()
        lexicon_path = os.path.join(settings.TREE_TAGGER, str(self.widget_id) + ".lex")
        if os.path.exists(lexicon_path):
            os.remove(lexicon_path)
        f = open(lexicon_path, 'a')
        cd = False
        for key in sorted(lexicon):
            tag_list = ""
            for i, tag in enumerate(lexicon[key]):
                if tag == 'CD':
                    if not cd:
                        if i == len(lexicon[key]) - 1:
                            tag_list += '\t' + tag + ' ' + '-\n'
                        else:
                            tag_list += '\t' + tag + ' ' + '-'
                        cd = True
                    else:
                        if tag_list and i == len(lexicon[key]) - 1:
                            tag_list += '\n'

                else:
                    if i == len(lexicon[key]) - 1:
                        tag_list += '\t' + tag + ' ' + '-\n'
                    else:
                        tag_list += '\t' + tag + ' ' + '-'
            if tag_list:
                f.write(key.encode('utf8')  + tag_list.encode('utf8'))
        f.close()
        unknown_words = os.path.join(settings.TREE_TAGGER, "tree_tagger_unknown_words.txt")

        param_path = os.path.join(self.params, 'english-utf8.par')
        pretrained_param = os.path.join(self.params, 'english-utf8-pretrained.par')
        if not os.path.exists(pretrained_param):
            os.rename(param_path, pretrained_param)
        else:
            os.remove(param_path)

        p = Popen([self._train_treetagger_bin, lexicon_path, unknown_words, train_path, param_path], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        a = p.communicate()


    def tag(self, sentences):
        """Tags a single sentence: a list of words.
        The tokens should not contain any newline characters.
        """
        encoding = self._encoding
        old = sentences
        # Write the actual sentences to the temporary input file
        if not self.trained:
            param_path = os.path.join(self.params, 'english-utf8.par')
            pretrained_param = os.path.join(self.params, 'english-utf8-pretrained.par')
            if os.path.exists(pretrained_param):
                os.remove(param_path)
                os.rename(pretrained_param, param_path)

        if isinstance(sentences, list):
            new_sents = []
            for sentence in sentences:
                sentence = ' '.join(sentence)
                sentence = sentence.replace(" 's ", "'s ")
                sentence = sentence.replace(" 'm ", "'m ")
                sentence = sentence.replace(" 're ", "'re ")
                sentence = sentence.replace(" 'll ", "'ll ")
                sentence = sentence.replace(" 've ", "'ve ")
                sentence = sentence.replace(" 'd ", "'d ")
                #sentence = sentence.replace('\xe2\x84\x87"', "\xe2\x84\x87")
                #sentence = re.sub(r" ([^ ']+)' ", r" \1 ", sentence)
                #sentence = re.sub(r" '([^ ']+) ", r" \1 ", sentence)
                #sentence = re.sub(r" ([^ ']+)-- ", r" \1 ", sentence)
                #sentence = re.sub(r" ([^ .]+)\. ", r" \1 ", sentence)

                new_sents.append(sentence)

            sentences = new_sents
            sentences = ' ### '.join(sentences)
           
            f = open(os.path.join(settings.TREE_TAGGER, str(self.widget_id) + ".txt"), 'w')
            f.write(sentences)
            f.close()
               
            p = Popen([self._treetagger_bin, os.path.join(settings.TREE_TAGGER, str(self.widget_id) + ".txt")], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                
                
            (taggedSnt, stderr) = p.communicate()
            
            sentences = []
            sentence = []
            splitted_tags = taggedSnt.split('\n')
            last = len(splitted_tags) - 1
            for i, line in enumerate(splitted_tags):
                line = line.strip()
                try:
                    word = line.split('\t')[0].strip()
                    tag = line.split('\t') [1].strip()

                    if word == '###':
                        sentences.append(sentence)
                        sentence = []
                    else:
                        if tag == 'SENT':
                            sentence.append((word, '.'))
                        else:
                            if tag in interpelations:
                                sentence.append((word, interpelations[tag]))
                            else:
                                sentence.append((word, tag))
                except:
                    pass
            sentences.append(sentence)
        checked_sentences = []
        counter = 0
        for i, sentence in enumerate(sentences):
            if len(old[i]) != len(sentence):
                counter += 1
                sentence = self.fixTokens(old[i], sentence)
            checked_sentences.append(sentence)
        print('counter: ', counter, len(sentences))
        return checked_sentences

    def fixTokens(self, standard, new):
        fixed = []
        fix = 0
        new_tag = ""
        if len(standard) < len(new):
            new_token=""
            for i, token in enumerate(new):
                word = token[0]
                tag = token[1]
                if word == standard[fix]:
                    fix += 1
                    fixed.append(token)
                else:
                    new_token += word
                    if len(new_tag) == 0 or len(word) > 1:
                        new_tag = tag
                    if new_token == standard[fix]:
                        fixed.append((new_token, tag))
                        fix += 1
                        new_token = ""
                        new_tag = ""
            if not len(fixed) == len(standard):
                print('ni popravu: ', standard, new)

            return fixed
        else:
            print('neki cudnga: ', standard, new)





    



