from mcdonals import McDonalsAssessment

from sklearn.feature_extraction.text import CountVectorizer

mc = McDonalsAssessment()

mc.task_7_init()

# Default
print("#################### DEFAULT TOKENISER ###############")
vect = CountVectorizer()
mc.tokenizer_perfomance(vect, "default")

# The results are slightly better than the default classifier from task6
# #################### DEFAULT TOKENISER ###############
# The AUC is 0.842807184867
# The Accuracy is 0.785326086957
# Number of Features: 7303

# Let"s try the best classifier that we found in task 6
print("################### STOPWORDS + MIN FRECUENCY ###############")
vect2 = CountVectorizer(stop_words="english", min_df=2)
mc.tokenizer_perfomance(vect2, "StopWords + MinFrecuency")

# Slightly worse classifier than the one in task6
# ################### STOPWORDS + MIN FRECUENCY ###############
# The AUC is 0.854188523287
# The Accuracy is 0.79347826087
# Number of Features: 3246

print("################### STOPWORDS ###############")
vect3 = CountVectorizer(stop_words="english")
mc.tokenizer_perfomance(vect3, "StopWords")

# ################### STOPWORDS ###############
# The AUC is 0.853234779844
# The Accuracy is 0.785326086957
# Number of Features: 7023

print("################### STOPWORDS & MAX & MIN FRECUENCY ###############")
vect9 = CountVectorizer(stop_words="english", min_df=4, max_df=0.3)
mc.tokenizer_perfomance(vect9, "StopWords + MinFrecuency + MaxFrecuency")

# ################### STOPWORDS & MAX & MIN FRECUENCY ###############
# The AUC is 0.864854554125
# The Accuracy is 0.807065217391
# Number of Features: 1739
