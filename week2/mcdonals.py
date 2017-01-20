import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


class NLP:
    def get_dtm_matrix(self, vect, X_train, X_test):
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)

        return X_train_dtm, X_test_dtm

    def model_performance(self, model, X_train_dtm, y_train, X_test_dtm, y_test):
        model.fit(X_train_dtm, y_train)

        # Calculate the probabilities, and keep only the prob of predicting Rude(1)
        y_pred_prob = model.predict_proba(X_test_dtm)[:, 1]

        # The Area Under the Curve (AUC)
        print("The AUC is {}".format(roc_auc_score(y_test, y_pred_prob)))

        # The Model Accuracy
        y_pred = model.predict(X_test_dtm)
        print("The Accuracy is {}".format(accuracy_score(y_test, y_pred)))

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

        return fpr, tpr

    def plot_roc_curve(self, data, plot_index=1, plot_label=None):
        plt.figure(plot_index)
        plt.plot([0, 1], [0, 1], 'k--')
        for model in data:
            label = plot_label if plot_label else model['label']
            plt.plot(model['fpr'], model['tpr'], label=label)

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        # Safe the plot with the last label
        plt.savefig('{}.png'.format(label))


class McDonalsAssessment(NLP):
    def __init__(self):
        # Read the data and clean it
        data = pd.read_csv('../data/mcdonalds.csv')
        data = data.dropna(subset=['policies_violated'], how='all')

        # Add a binary rude colum
        data['rude'] = data.policies_violated.str.contains('RudeService').astype(int)

        # Define our model data
        X = data.review
        y = data.rude

        # Split the data into training and testing data
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(X, y, random_state=1)

        assert self.X_train.shape == self.y_train.shape
        assert self.X_test.shape == self.y_test.shape

    def task_7_init(self):
        # Read the data and clean it
        data = pd.read_csv('../data/mcdonalds.csv')
        data = data.dropna(subset=['policies_violated'], how='all')

        # Add a binary rude colum
        data['rude'] = data.policies_violated.str.contains('RudeService').astype(int)

        # Clean City column from NaN values
        data.city = data.city.fillna("na")

        # Define our model data - review contatenated with the city from the review
        X = data.review.str.cat(data.city, sep=" ")
        y = data.rude

        # Split the data into training and testing data
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(X, y, random_state=1)

        assert self.X_train.shape == self.y_train.shape
        assert self.X_test.shape == self.y_test.shape

    def tokenizer_perfomance(self, vect, plot_index, plot_label=None):
        """
        Using a simple Multinomial Naive Bayes check some performance
        """
        X_train_dtm, X_test_dtm = self.get_dtm_matrix(vect, self.X_train, self.X_test)

        nb = MultinomialNB()
        fpr, tpr = self.model_performance(
            nb,
            X_train_dtm,
            self.y_train,
            X_test_dtm,
            self.y_test
        )

        print("Number of Features: {}".format(X_train_dtm.shape[1]))

        data = {
            'fpr': fpr,
            'tpr': tpr,
            'label': "MultinomialNB",
        },

        self.plot_roc_curve(data, plot_index=plot_index, plot_label=plot_label)
