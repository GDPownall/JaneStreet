#!/usr/bin/env python

lines = [x.strip() for x in open('input/features.csv','r')]
features = {}
for l in lines:
    if 'tag' in l: continue
    l = l.split(',')
    features[l[0]] = ','.join(l[1:])

rev_multidict = {}
for key, value in features.items():
    rev_multidict.setdefault(value, set()).add(key)

print [values for key, values in rev_multidict.items() if len(values) > 1]
