from collections import Iterable
def flat(lis):
     for item in lis:
         if isinstance(item, list):# and not isinstance(item, basestring):
             for x in flat(item):
                 yield x
         else:
             yield item

def flatten(lis):
    return list(flat(lis))