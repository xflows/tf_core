from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from nltoolkit.helpers import TokenSplitter

from tf_core.nltoolkit.helpers import TokenSplitter


class BowModelConstructor:
    def __init__(self,adc,token_annotation,stem_feature_name,stop_word_feature_name,
                 label_doc_feature_name,
                weighting_type, normalize_vectors,
                 max_ngram,min_tf,predefined_vocabulary=None):

        self._feature_name=token_annotation+('/'+stem_feature_name if stem_feature_name else '')

        self._stop_word_feature_name=stop_word_feature_name
        self._doc_class_label=label_doc_feature_name

        raw_documents=self.get_raw_text(adc.documents,join_annotations_with='|##|')
        #extract vocabulary
        #document_tokens=self.get_annotation_texts(adc.documents)

        # vocabulary={}
        # for raw_doc in document_tokens:
        #     for token in raw_doc:
        #         if not token in vocabulary:
        #             vocabulary[token]=len(vocabulary)

        #(?u)\\b\\w+\\b
        self.__count_params={'ngram_range':(1,max_ngram),'min_df':min_tf,'tokenizer':TokenSplitter(),
                             'token_pattern':'\\b\\w+\\b'} #'vocabulary':vocabulary}
        #tf_idf args: 'norm', 'smooth_idf', 'sublinear_tf', 'use_idf
        self.__tfidf_params=self.__wighting_type_to_tfidf_params(weighting_type)
        self.__tfidf_params['norm']='l2' if normalize_vectors else None

        #set default vectorizerl
        self.vectorizer =self._count_vectorizer() if weighting_type=='term_freq' else self._tfidf_vectorizer() #DictVectorizer(dtype=dtype, sparse=sparse)

        #set predefined controlled vocabulary
        if predefined_vocabulary:
            vocab_vectorizer=CountVectorizer(ngram_range=(1,max_ngram),tokenizer=lambda doc: [doc]) #every line represents a new token
            vocab_vectorizer.fit(predefined_vocabulary)
            self.set_new_vocabulary(vocab_vectorizer.vocabulary_,raw_documents)
        else:
            self.vectorizer.fit(raw_documents) #fit the vectorizer to the documents
            self.__count_params['vocabulary']=self.vectorizer.vocabulary_ #set the learned vocabulary also to future vectorizers

        #print self.vectorizer.get_feature_names()

    def set_new_vocabulary(self,vocabulary,raw_documents,intersect=True):
        if intersect: #intersect vocabularies
            self.vectorizer.fit(raw_documents) #fit the vectorizer to the documents
            feature_intersection=[term for term in self.vectorizer.get_feature_names()
                                  if term in vocabulary]

            self.vectorizer.set_params(vocabulary=dict([(a,i) for i,a in enumerate(feature_intersection)])) #set new vocabulary
            self.vectorizer.fit(raw_documents)
            self.__count_params['vocabulary']=self.vectorizer.vocabulary_ #set the learned vocabulary also to future vectorizers

        else:
            self.__count_params['vocabulary']=vocabulary


    @staticmethod
    def __wighting_type_to_tfidf_params(weighting_type):
        if weighting_type=='term_freq':
            return {}
        elif weighting_type=='tf_idf':
            return {'use_idf':True,'smooth_idf':False,'sublinear_tf':False}
        elif weighting_type=='tf_idf_safe':
            return {'use_idf':True,'smooth_idf':True,'sublinear_tf':False}
        elif weighting_type=='log_df_tf_idf':
            return {'use_idf':True,'smooth_idf':False,'sublinear_tf':True}


    def _vocab_to_idx(self):
        return self.__count_params['vocabulary']


    def _idx_to_vocab(self):
        return {v: k for k, v in self._vocab_to_idx().items()}


    def _count_vectorizer(self):
        return CountVectorizer(**self.__count_params)
    def _tfidf_vectorizer(self):
        return TfidfVectorizer(**dict(self.__count_params.items()
                                      +self.__tfidf_params.items()
                                      #+[['tokenizer',self.custom_tokenizer]]
                                      ))
    def _tfidf_transformer(self):
        return TfidfTransformer(**self.__tfidf_params)



    def get_raw_text(self,documents,join_annotations_with=" "):
        return [document.raw_text(selector=self._feature_name,
                                  join_annotations_with=join_annotations_with,
                                  stop_word_feature_name=self._stop_word_feature_name)
                for document in documents]

    def get_annotation_texts(self,documents):
        return [document.get_annotation_texts(selector=self._feature_name,
                                  stop_word_feature_name=self._stop_word_feature_name)
                for document in documents]


    def get_document_labels(self,adc,binary=False):
        '''
        :param adc: annotated document corpus
        :param binary: return binary results
        :return: for every document if it contains the selected label
        '''

        if self._doc_class_label:
            res=[doc.get_first_label(self._doc_class_label) for doc in adc.documents]
            if binary:
                uniq_res=list(set(res))
                return [uniq_res.index(r) for r in res]
            return res
        else:
            return None
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

#BowSpace
        # private ITokenizer mTokenizer
        #     = new UnicodeTokenizer();
        # private Set<string>.ReadOnly mStopWords
        #     = null;
        # private IStemmer mStemmer
        #     = null;
        # private Dictionary<string, Word> mWordInfo
        #     = new Dictionary<string, Word>();
        # private ArrayList<Word> mIdxInfo
        #     = new ArrayList<Word>();
        # private int mMaxNGramLen
        #     = 2;
        # private int mMinWordFreq
        #     = 5;
        # private WordWeightType mWordWeightType
        #     = WordWeightType.TermFreq;
        # private double mCutLowWeightsPerc
        #     = 0.2;
        # private bool mNormalizeVectors
        #     = true;
        # private bool mKeepWordForms
        #     = false;
