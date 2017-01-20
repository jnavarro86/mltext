from __future__ import print_function

from mcdonals import McDonalsAssessment

from sklearn.feature_extraction.text import CountVectorizer

mc = McDonalsAssessment()

# Default
print("#################### DEFAULT TOKENISER ###############")
vect = CountVectorizer()
mc.tokenizer_perfomance(vect, "default")

# lowercase
print("#################### LOWERCASE ###############")
vect2 = CountVectorizer(lowercase=False)
mc.tokenizer_perfomance(vect2, "lowercase")

# ngrames
print("#################### 2 NGRAMES ###############")
vect3 = CountVectorizer(ngram_range=(1, 2))
mc.tokenizer_perfomance(vect3, "2 ngrames")

# stop_words
print("#################### ENGLISH STOPWORDS ###############")
vect4 = CountVectorizer(stop_words="english")
mc.tokenizer_perfomance(vect4, "English StopWords")

# max_features
print("#################### 1000 FEATURES MAX ###############")
vect5 = CountVectorizer(max_features=1000)
mc.tokenizer_perfomance(vect5, "Max Features")

# Max Document frecuency
print("#################### MAX DOCUMENT FRECUENCY ###############")
vect6 = CountVectorizer(max_df=0.5)
mc.tokenizer_perfomance(vect6, "Max Frecuency")

# Min Document Frecuency
print("################### MIN DOCUMENT FRECUENCY ###############")
vect7 = CountVectorizer(min_df=2)
mc.tokenizer_perfomance(vect7, "Min Frecuency")

# The results are
# #################### DEFAULT TOKENISER ###############
# The AUC is 0.842600540455
# The Accuracy is 0.798913043478
# Number of Features: 7300
# #################### LOWERCASE ###############
# The AUC is 0.840645366396
# The Accuracy is 0.798913043478
# Number of Features: 8742
# #################### 2 NGRAMES ###############
# The AUC is 0.819599427754
# The Accuracy is 0.766304347826
# Number of Features: 57936
# #################### ENGLISH STOPWORDS ###############
# The AUC is 0.853520902877
# The Accuracy is 0.798913043478
# Number of Features: 7020
# #################### 1000 FEATURES MAX ###############
# The AUC is 0.830090605627
# The Accuracy is 0.779891304348
# Number of Features: 1000
# #################### MAX DOCUMENT FRECUENCY ###############
# The AUC is 0.844889524718
# The Accuracy is 0.788043478261
# Number of Features: 7291
# ################### MIN DOCUMENT FRECUENCY ###############
# The AUC is 0.844587505961
# The Accuracy is 0.790760869565
# Number of Features: 3500

# After this results, we can see how for example reducing the number of features
# drastically to 1000 didn"t influcence the performance that much. Let"s see if we clean
# improve our best performance (.85) combining some of the techniques.

print("################### MIN FRECUENCY + MAX FEATURES ###############")
vect8 = CountVectorizer(min_df=2, max_features=2000)
mc.tokenizer_perfomance(vect8, "MinF + MaxFeatures")

# ################### MIN FRECUENCY + MAX FEATURES ###############
# The AUC is 0.839214751232
# The Accuracy is 0.790760869565
# Number of Features: 2000

print("################### STOPWORDS + MIN FRECUENCY ###############")
vect9 = CountVectorizer(stop_words="english", min_df=2)
mc.tokenizer_perfomance(vect9, "StopWords + MinFrecuency")

# This is the best classifier so far
# ################### STOPWORDS + MIN FRECUENCY ###############
# The AUC is 0.85447464632
# The Accuracy is 0.79347826087
# Number of Features: 3241
