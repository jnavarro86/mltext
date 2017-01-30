from mcdonals import McDonalsAssessment

from sklearn.feature_extraction.text import CountVectorizer

mc = McDonalsAssessment()

mc.task_8_init()

# Best Model
vect = CountVectorizer(stop_words="english", min_df=4, max_df=0.3)
mc.tokenizer_perfomance(vect, "task_8_BestModel")

# The AUC is 0.849546971865
# The Accuracy is 0.804347826087
# Number of Features: 1344
