from __future__ import print_function

import numpy as np

from mcdonals import McDonalsAssessment

from sklearn.feature_extraction.text import CountVectorizer

mc = McDonalsAssessment()

# # Default
# print("#################### DEFAULT TOKENISER ###############")
# vect = CountVectorizer()
# mc.tokenizer_perfomance(vect, "default")
#
# # lowercase
# print("#################### LOWERCASE ###############")
# vect2 = CountVectorizer(lowercase=False)
# mc.tokenizer_perfomance(vect2, "lowercase")
#
# # ngrames
# print("#################### 2 NGRAMES ###############")
# vect3 = CountVectorizer(ngram_range=(1, 2))
# mc.tokenizer_perfomance(vect3, "2 ngrames")
#
# # stop_words
# print("#################### ENGLISH STOPWORDS ###############")
# vect4 = CountVectorizer(stop_words="english")
# mc.tokenizer_perfomance(vect4, "English StopWords")
#
# # max_features
# print("#################### 1000 FEATURES MAX ###############")
# vect5 = CountVectorizer(max_features=1000)
# mc.tokenizer_perfomance(vect5, "Max Features")
#
# # Study of the impact of max_df in the AUC
# # for max_df in np.arange(0.1, 1, 0.1):
# #     print("Min Document Frecuency {}".format(max_df))
# #     vect5 = CountVectorizer(max_df=max_df)
# #     print(mc.get_auc(vect5))
#
# # We can see that the trend is using a max of 0.3
#
# # Max Document frecuency
# print("#################### MAX DOCUMENT FRECUENCY ###############")
# vect6 = CountVectorizer(max_df=0.3)
# mc.tokenizer_perfomance(vect6, "Max Frecuency")
#
# # # Study of the impact of min_frec in the AUC
# # for min_frec in xrange(1, 20):
# #     print("Min Document Frecuency {}".format(min_frec))
# #     vect7 = CountVectorizer(min_df=min_frec)
# #     print(mc.get_auc(vect7))
#
# # The AUC starts with 0.84, then drops a little bit, then reach its max with min_df in 2
# # and from that drops slowly until 0.75.
#
# # Min Document Frecuency
# print("################### MIN DOCUMENT FRECUENCY ###############")
# vect7 = CountVectorizer(min_df=2)
# mc.tokenizer_perfomance(vect7, "Min Frecuency")
#
#
# #
# # # The results are
# # #################### DEFAULT TOKENISER ###############
# # The AUC is 0.842600540455
# # The Accuracy is 0.798913043478
# # Number of Features: 7300
# # #################### LOWERCASE ###############
# # The AUC is 0.840645366396
# # The Accuracy is 0.798913043478
# # Number of Features: 8742
# # #################### 2 NGRAMES ###############
# # The AUC is 0.819599427754
# # The Accuracy is 0.766304347826
# # Number of Features: 57936
# # #################### ENGLISH STOPWORDS ###############
# # The AUC is 0.853520902877
# # The Accuracy is 0.798913043478
# # Number of Features: 7020
# # #################### 1000 FEATURES MAX ###############
# # The AUC is 0.830090605627
# # The Accuracy is 0.779891304348
# # Number of Features: 1000
# # #################### MAX DOCUMENT FRECUENCY ###############
# # The AUC is 0.852376410746
# # The Accuracy is 0.804347826087
# # Number of Features: 7269
# # ################### MIN DOCUMENT FRECUENCY ###############
# # The AUC is 0.844587505961
# # The Accuracy is 0.790760869565
# # Number of Features: 3500
#
# # After this results, we can see how for example reducing the number of features
# # drastically to 1000 didn"t influcence the performance that much. Let"s see if we clean
# # improve our best performance (.85) combining some of the techniques.
#
# print("################### MIN FRECUENCY + MAX FEATURES ###############")
# vect8 = CountVectorizer(min_df=2, max_features=2000)
# mc.tokenizer_perfomance(vect8, "MinF + MaxFeatures")
#
# # ################### MIN FRECUENCY + MAX FEATURES ###############
# # The AUC is 0.839214751232
# # The Accuracy is 0.790760869565
# # Number of Features: 2000
#
# print("################### STOPWORDS + MIN FRECUENCY ###############")
# vect9 = CountVectorizer(stop_words="english", min_df=2)
# mc.tokenizer_perfomance(vect9, "StopWords + MinFrecuency")
#
# # This is the best classifier so far
# # ################### STOPWORDS + MIN FRECUENCY ###############
# # The AUC is 0.85447464632
# # The Accuracy is 0.79347826087
# # Number of Features: 3241

print("################### STOPWORDS & MAX & MIN FRECUENCY ###############")
vect9 = CountVectorizer(stop_words="english", min_df=4, max_df=0.3)
mc.tokenizer_perfomance(vect9, "StopWords + MinFrecuency + MaxFrecuency")

# Best model found
# ################### MAX & MIN FRECUENCY ###############
# The AUC is 0.862152281036
# The Accuracy is 0.809782608696
# Number of Features: 1732
