# kaiburr_assessment
KAIBURR ASSESSMENT
TASK 6:
❖ Dataset
❖ Loading the data
❖ Feature Engineering
❖ Text processing
❖ Exploring Multi-classification Models
❖ Compare Model performance
❖ Evaluation
❖ Prediction
Our aim is to classify the complaints of the consumer into predefined categories using a suitable 
classification algorithm. For now, we will be using the following classification algorithms.
➢ Linear Support Vector Machine (LinearSVM)
➢ Random Forest
➢ Multinomial Naive Bayes
➢ Logistic Regression
First, we will install the required modules.
• import pandas as pd
• import numpy as np
• from scipy.stats import randint
• import seaborn as sns # used for plot interactive graph.
• import matplotlib.pyplot as plt
• import seaborn as sns
• from io import StringIO
• from sklearn.feature_extraction.text import TfidfVectorizer
• from sklearn.feature_selection import chi2
• from IPython.display import display
• from sklearn.model_selection import train_test_split
• from sklearn.feature_extraction.text import TfidfTransformer
• from sklearn.naive_bayes import MultinomialNB
• from sklearn.linear_model import LogisticRegression
• from sklearn.ensemble import RandomForestClassifier
• from sklearn.svm import LinearSVC
• from sklearn.model_selection import cross_val_score
• from sklearn.metrics import confusion_matrix
• from sklearn import metrics
I have implemented a basic Text classification model using a few algorithms and evaluated 
the model using accuracy,taken a few sample inputs and predicted.
