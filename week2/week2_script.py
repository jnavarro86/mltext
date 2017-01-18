from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Read the data and clean it
data = pd.read_csv('../data/mcdonalds.csv')
data = data.dropna(subset=['policies_violated'], how='all')

# Add a binary rude colum
data['rude'] = data.policies_violated.str.contains('RudeService').astype(int)

# Define our model data
X = data.review
y = data.rude

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
assert X_train.shape == y_train.shape
assert X_test.shape == y_test.shape

# Tokenise the inputs

# Initialise the vectorizer
vect = CountVectorizer()
# learn the words and transform the input into the document terms matrix
X_train_dtm = vect.fit_transform(X_train)
# Transform the test input data
X_test_dtm = vect.transform(X_test)

# Now train and test dtms should have the same number of features
assert X_train_dtm.shape[1] == X_test_dtm.shape[1]


def model_performance(model, X_train_dtm, y_train, X_test_dtm, y_test):
    model.fit(X_train_dtm, y_train)

    # Calculate the probabilities, and keep only the prob of predicting Rude(1)
    y_pred_prob = model.predict_proba(X_test_dtm)[:, 1]
    y_pred_log_prob = model.predict_log_proba(X_test_dtm)[:, 1]

    # The Area Under the Curve (AUC)
    print("The AUC is {}".format(roc_auc_score(y_test, y_pred_prob)))
    print(
        "The AUC using log probabilities is {}".format(
            roc_auc_score(y_test, y_pred_log_prob)
        )
    )

    # The Model Accuracy
    y_pred = model.predict(X_test_dtm)
    print("The Accuracy is {}".format(accuracy_score(y_test, y_pred)))

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    return fpr, tpr


def plot_roc_curve(data):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    for model in data:
        plt.plot(model['fpr'], model['tpr'], label=model['label'])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


# Test the performance of a Multinomial Naive Bayes
nb = MultinomialNB()
print("#### Multinomial Naive Bayes #####")
fpr_nb, tpr_nb = model_performance(nb, X_train_dtm, y_train, X_test_dtm, y_test)

# Test the performance of a Logistic Regression
lg = LogisticRegression()
print("#### Logistic Regression #####")
fpr_lg, tpr_lg = model_performance(lg, X_train_dtm, y_train, X_test_dtm, y_test)

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

plot_roc_curve(data)
import pdb; pdb.set_trace()
