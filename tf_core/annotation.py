

class Annotation:
    def __init__(self, span_start, span_end, type, features=None):
        self.features=features or {}
        self.span_start=span_start
        self.span_end=span_end
        self.type=type

        if span_start<0 or span_end<0 or span_start>span_end:
            raise Exception('Invalid span_start or span_end in %s' % str(self))
        
    def __repr__(self):
        return '<Annotation span_start:%d span_ned:%d>' % (self.span_start, self.span_end)

    def __unicode__(self):
        return 'span_start: %d, span_ned: %d' % (self.span_start, self.span_end)
    
    def __str__(self):
        return unicode(self).encode('utf-8')