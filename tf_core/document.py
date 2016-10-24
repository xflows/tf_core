import json
from collections import defaultdict

from tf_core.annotation import Annotation


class Document:
    def __init__(self, name,text,annotations,features):
        self.annotations=annotations
        self.features=features
        self.name=name

        self.text=text

    def __unicode__(self):
        return 'Name; {0}\nText: {1}' % (self.name, self.text)

    def __minimize__(self):
        state=self.__dict__.copy()
        annotations=state.pop('annotations')
        annotations_per_type=defaultdict(list)

        for ann in annotations:
            annotations_per_type[ann.type].append([ann.span_start,ann.span_end,ann.features])

        state['annotations']=dict(annotations_per_type)
        return state #json.dumps(state)

    @classmethod
    def __from_minimized__(cls,state):

        all_annotations=[]
        for ann_type,annotations in state['annotations'].items():
            for ann in annotations:
                all_annotations.append(Annotation(ann[0], ann[1], ann_type, ann[2]))
        return cls(state['name'],state['text'],all_annotations,state['features'])


    def get_annotations_with_text(self, selector):
        """
        :param selector: textual string in one of the following formats:
           a) annotation_name
           b) annotation_name/feature_name       so you can do for instance stopword tagging on lemmas
        :return: list of selected (annotation,text) tuples
        """
        annotations_with_text = []
        selector_split = selector.split("/")
        element_annotation = selector_split[0].strip()
        element_feature = False if len(selector_split) == 1 else selector_split[1].strip()

        for a in self.annotations:
            if a.type == element_annotation:
                try:
                    if element_feature:
                            text = a.features[element_feature]
                    else:
                        text = self.text[a.span_start:a.span_end+1]
                    annotations_with_text.append((a, text))
                except KeyError:
                     #raise KeyError("The Annotation (%s) does not have feature named '%s'!" % (a.__str__(), element_feature))
                     pass
        return annotations_with_text
    def get_annotations(self, selector):
        return [a[0] for a in self.get_annotations_with_text(selector)]

    def get_annotation_texts(self,selector,stop_word_feature_name="StopWord"):
        return [text for (ann,text) in self.get_annotations_with_text(selector)
                               if not ann.features.has_key(stop_word_feature_name)]

    def raw_text(self,selector=None,stop_word_feature_name="StopWord",join_annotations_with=" "):
        if not selector:
            return self.text
        else:
            selected_subtexts=self.get_annotation_texts(selector,stop_word_feature_name)
            return join_annotations_with.join(selected_subtexts)

    def get_first_label(self,label_feature_name="Labels"):
        label_value=self.features.get(label_feature_name,None)
        #for klass in classes:
        #    if klass in self.features:
        #        return klass
        #d=444
        if label_value==None:
            return ''
        try:
            #print label_value
            true_value=json.loads(label_value)
            return true_value[0] if type(true_value)==list else label_value #if not a json list
        except ValueError, e:
            return label_value