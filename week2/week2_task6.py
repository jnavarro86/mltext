from __future__ import print_function

from mcdonals import McDonalsAssessment

from sklearn.feature_extraction.text import CountVectorizer

mc = McDonalsAssessment()

# Default
print("#################### DEFAULT TOKENISER ###############")
vect = CountVectorizer()
mc.tokenizer_perfomance(vect, 1, "default")

# lowercase
print("#################### LOWERCASE ###############")
vect2 = CountVectorizer(lowercase=False)
mc.tokenizer_perfomance(vect2, 2, "lowercase")

# ngrames
print("#################### 2 NGRAMES ###############")
vect3 = CountVectorizer(ngram_range=(1, 2))
mc.tokenizer_perfomance(vect3, 3, "2 ngrames")

# stop_words
print("#################### ENGLISH STOPWORDS ###############")
vect4 = CountVectorizer(stop_words="english")
mc.tokenizer_perfomance(vect4, 4, "English StopWords")
