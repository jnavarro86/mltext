from __future__ import print_function

from mcdonals import McDonalsAssessment
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


mc = McDonalsAssessment()
# Tokenise the inputs
# Initialise the vectorizer
vect = CountVectorizer()

# learn the words and transform the input into the document terms matrix
X_train_dtm = vect.fit_transform(mc.X_train)
# Transform the test input data
X_test_dtm = vect.transform(mc.X_test)

# Now train and test dtms should have the same number of features
assert X_train_dtm.shape[1] == X_test_dtm.shape[1]

# Test the performance of a Multinomial Naive Bayes
nb = MultinomialNB()
print("#### Multinomial Naive Bayes #####")
fpr_nb, tpr_nb = mc.model_performance(nb, X_train_dtm, mc.y_train, X_test_dtm, mc.y_test)

# Test the performance of a Logistic Regression
lg = LogisticRegression()
print("#### Logistic Regression #####")
fpr_lg, tpr_lg = mc.model_performance(lg, X_train_dtm, mc.y_train, X_test_dtm, mc.y_test)

data = [
    {
        'fpr': fpr_nb,
        'tpr': tpr_nb,
        'label': "MultinomialNB",
    },
    {
        'fpr': fpr_lg,
        'tpr': tpr_lg,
        'label': "LogisticRegression",
    }
]

mc.plot_roc_curve(data)
