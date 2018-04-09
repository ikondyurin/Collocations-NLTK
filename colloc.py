import nltk
from nltk.collocations import *
from nltk.metrics import BigramAssocMeasures
import operator
import string
import math

#tokenisation
l = []
with open("book.txt", 'r', encoding="utf-8") as f:
    for line in f:
        for w in nltk.word_tokenize(line.lower()):
            if w not in string.punctuation:
                l.append(w)
#frequency list
freq = nltk.FreqDist(l)
#sorting
sort_fr = freq.most_common()

#writing into file
with open("freq.csv", 'w', encoding="windows-1251") as freq_d:
    for i, pair in enumerate(sort_fr):
        freq_d.write("{}; {}; {}; {}; {}\n".format(pair[0], pair[1], i+1, round(math.log1p(pair[1]),2), round(math.log1p(i+1),2)))

#building lists of collocations using four measures
measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(l)
t = finder.nbest(measures.student_t, 100)
c = finder.nbest(measures.chi_sq, 100)
l = finder.nbest(measures.likelihood_ratio, 100)
p = finder.nbest(measures.pmi, 100)

#
#
#
#
#
#to observe the problem try this:
"""
for q in p:
    print(q)
    print(finder.score_ngram(measures.pmi, n[0], n[1]))
"""

#writing collocations and values into .csv file
with open("collocations.csv", 'w', encoding="windows-1251") as coll:
    for i in t:
        coll.write("{}; {}; {}\n".format("t", i[0]+" "+i[1], round(finder.score_ngram(measures.student_t, i[0], i[1]),2)))
    for m in t:
        coll.write("{}; {}; {}\n".format("chi^2", m[0]+" "+m[1], round(finder.score_ngram(measures.chi_sq, m[0], m[1]),2)))
    for n in t:
        coll.write("{}; {}; {}\n".format("log-likelihood", n[0]+" "+n[1], round(finder.score_ngram(measures.likelihood_ratio, n[0], n[1]),2)))
    for q in t:
        coll.write("{}; {}; {}\n".format("pmi", q[0]+" "+q[1], round(finder.score_ngram(measures.pmi, q[0], q[1]),2)))
      

