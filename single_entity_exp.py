import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, accuracy_score 
from sklearn.preprocessing import LabelEncoder 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier
import joblib
import glob 
import os 
from sklearn.pipeline import Pipeline 
import mlflow 
import mlflow.sklearn 



sent_analyzer = SentimentIntensityAnalyzer()
path = r'C:\Users\PATRICK\Desktop\MLAI\brand_comparison'
raw_files = glob.glob(os.path.join(path,  "*.csv"))
dfs_list = []
for file in raw_files: 
    df = pd.read_csv(file)
    dfs_list.append(df)
all_tweets_df = pd.concat(dfs_list, axis=0, ignore_index=True)
#print(all_tweets_df.head(2))
all_tweets_df['compound_score'] = all_tweets_df['clean_tweets'].apply(lambda x: sent_analyzer.polarity_scores(str(x))['compound'])
all_tweets_df['label'] = all_tweets_df['compound_score'].apply(lambda x: "Positive" if x > 0.0 else "Negative" if x < 0.0 else "Neutral")
all_tweets_df = all_tweets_df.sample(frac=1) #randomly shuffling the tweets to ensure randomness of the 
#print(all_tweets_df.head(2))

all_tweets_df = all_tweets_df.dropna()
#print(all_tweets_df.info())
#print(all_tweets_df['label'].unique())
'''
#Feature Engineering Experiments:

1. Count vectorizers 
2. Term-inverse frequency 
3. Weighted vectors. 
'''
#initializing the vectorizers to be used  
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

#fitting both vectorizers to the training set
count_vectorizer.fit(all_tweets_df['clean_tweets']), tfidf_vectorizer.fit(all_tweets_df['clean_tweets'])
#splitting data into train test splits 
X_train, X_test, y_train, y_test = train_test_split(all_tweets_df['clean_tweets'], all_tweets_df['label'], test_size=0.3, random_state=42)
#converting text to vectors
X_train_cv = count_vectorizer.transform(X_train)
X_test_cv =  count_vectorizer.transform(X_test)
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#Experimenting With Different Models
lr = LogisticRegression()
dtree =DecisionTreeClassifier()
rf =RandomForestClassifier()
nnet = MLPClassifier()
svc = SVC()

models = [lr, dtree, rf, nnet, svc]
model_acc, model_f1 = [], []
for model in models:
    model.fit(X_train_cv, y_train)
    model_acc = accuracy_score(model.predict(X_test_cv),y_test)
    model_f1 = f1_score(model.predict(X_test_cv), y_test, average="weighted")
print(model_acc)
#Logging Accuracies
mlflow.log_metric("Log Reg Acc", model_acc[0])
mlflow.log_metric("Decision Tree Acc", model_acc[1])
mlflow.log_metric("Random forest Acc", model_acc[2])
mlflow.log_metric("Neural Net Acc", model_acc[3])
mlflow.log_metric("Support vector Acc", model_acc[4])

#Logging F1 scores
mlflow.log_metric("Log Reg F1", model_f1[0])
mlflow.log_metric("Decision Tree F1", model_f1[1])
mlflow.log_metric("Random forest F1", model_f1[2])
mlflow.log_metric("Neural Net F1", model_f1[3])
mlflow.log_metric("Support vector F1", model_f1[4])

#Logging the models used
mlflow.sklearn.log_model("lr", model)
mlflow.sklearn.log_model("dtree", model)
mlflow.sklearn.log_model("rf", model)
mlflow.sklearn.log_model("nnet", model)
mlflow.sklearn.log_model("svc", model)