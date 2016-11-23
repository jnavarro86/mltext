## Machine Learning with Text in Python

This repo contains the solutions for the assessments proposed in the following course - 

* [Course information and FAQs](http://www.dataschool.io/learn/)
* **Instructor:** [Kevin Markham](http://www.dataschool.io/about/)
* **Teaching Assistant:** [Alex Egorenkov](https://www.linkedin.com/in/aegorenkov)

### Course Schedule

* [Before the Course](#before-the-course)
* [Week 1: Working with Text Data in scikit-learn](#week-1-working-with-text-data-in-scikit-learn)
* [Week 2: Basic Natural Language Processing (NLP)](#week-2-basic-natural-language-processing-nlp)
* [Week 3: Intermediate NLP and Basic Regular Expressions](#week-3-intermediate-nlp-and-basic-regular-expressions)
* [Week 4: Intermediate Regular Expressions](#week-4-intermediate-regular-expressions)
* [Week 5: Working a Text-Based Data Science Problem](#week-5-working-a-text-based-data-science-problem)
* [Week 6: Advanced Machine Learning Techniques](#week-6-advanced-machine-learning-techniques)
* [After the Course](#after-the-course)

-----

### Before the Course

* Make sure that [scikit-learn](http://scikit-learn.org/stable/install.html), [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html), and [matplotlib](http://matplotlib.org/users/installing.html) (and their dependencies) are installed on your system. The easiest way to accomplish this is by downloading the [Anaconda distribution](https://www.continuum.io/downloads) of Python. Both Python 2 and 3 are welcome.
* If you are not familiar with Git and GitHub, watch my [quick introduction to Git and GitHub](https://www.youtube.com/watch?v=zYG8B8q722g) (8 minutes). Note that the repository shown in the video is from a previous iteration of the course, and the GitHub interface has also changed slightly.
    * For a longer introduction to Git and GitHub, watch my [11-video series](https://www.youtube.com/playlist?list=PL5-da3qGB5IBLMp7LtN8Nc3Efd4hJq0kD) (36 minutes).
* If you are not familiar with the Jupyter notebook, watch my [introductory video](https://www.youtube.com/watch?v=IsXXlYVBt1M&t=4m57s) (8 minute segment). Note that the Jupyter notebook was previously called the "IPython notebook", and the interface has also changed slightly. (Here is the [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/02_machine_learning_setup.ipynb) shown in the video.)
* If you are not yet comfortable with scikit-learn, review the notebooks and/or videos from my [scikit-learn video series](https://github.com/justmarkham/scikit-learn-videos), focusing specifically on the following topics:
    * Machine learning terminology, and working with data in scikit-learn ([video 3](https://www.youtube.com/watch?v=hd1W4CyPX58&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=3), [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/03_getting_started_with_iris.ipynb))
    * scikit-learn's 4-step modeling pattern ([video 4](https://www.youtube.com/watch?v=RlQuVL6-qe8&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=4), [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/04_model_training.ipynb))
    * Train/test split ([video 5](https://www.youtube.com/watch?v=0pP4EwWJgIU&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=5), [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/05_model_evaluation.ipynb))
    * Accuracy, confusion matrix, and AUC ([video 9](https://www.youtube.com/watch?v=85dtiMz9tSo&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=9), [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb))
* If you are not yet comfortable with pandas, review the notebook and/or videos from my [pandas video series](https://github.com/justmarkham/pandas-videos). Alternatively, review another one of my [recommended pandas resources](http://www.dataschool.io/best-python-pandas-resources/).

-----

### Week 1: Working with Text Data in scikit-learn

**Topics covered:**
* Model building in scikit-learn (refresher)
* Representing text as numerical data
* Reading the SMS data
* Vectorizing the SMS data
* Building a Naive Bayes model
* Comparing Naive Bayes with logistic regression
* Calculating the "spamminess" of each token
* Creating a DataFrame from individual text files

-----

### Week 2: Basic Natural Language Processing (NLP)

**Topics covered:**
* What is NLP?
* Reading in the Yelp reviews corpus
* Tokenizing the text
* Comparing the accuracy of different approaches
* Removing frequent terms (stop words)
* Removing infrequent terms
* Handling Unicode errors
-----

### Week 3: Intermediate NLP and Basic Regular Expressions

**Topics covered:**
* Intermediate NLP:
    * Reading in the Yelp reviews corpus
    * Term Frequency-Inverse Document Frequency (TF-IDF)
    * Using TF-IDF to summarize a Yelp review
    * Sentiment analysis using TextBlob
* Basic Regular Expressions:
    * Why learn regular expressions?
    * Rules for searching
    * Metacharacters
    * Quantifiers
    * Using regular expressions in Python
    * Match groups
    * Character classes
    * Finding multiple matches

-----

### Week 4: Intermediate Regular Expressions

**Topics covered:**
* Week 3 homework review
* Greedy or lazy quantifiers
* Alternatives
* Substitution
* Anchors
* Option flags
* Assorted functionality

-----

### Week 5: Working a Text-Based Data Science Problem

**Topics covered:**
* Reading in and exploring the data
* Feature engineering
* Model evaluation using train_test_split and cross_val_score
* Making predictions for new data
* Searching for optimal tuning parameters using GridSearchCV
* Extracting features from text using CountVectorizer
* Chaining steps into a Pipeline

-----

### Week 6: Advanced Machine Learning Techniques

**Topics covered:**
* Reading in the Kaggle data and adding features
* Using a Pipeline for proper cross-validation
* Combining GridSearchCV with Pipeline
* Efficiently searching for tuning parameters using RandomizedSearchCV
* Adding features to a document-term matrix (using SciPy)
* Adding features to a document-term matrix (using FeatureUnion)
* Ensembling models
* Locating groups of similar cuisines
* Model stacking

-----
