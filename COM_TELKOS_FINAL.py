from optparse import Values
import joblib
import tweepy
from tweepy import OAuthHandler
import pandas as pd
import numpy as np
import re
import nltk
nltk.download("stopwords")
import spacy 
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from wordcloud import WordCloud, STOPWORDS
import ssl 
import smtplib
from email.message import EmailMessage 
from pathlib import Path 
import os
from dotenv import load_dotenv
import sqlite3
from datetime import datetime


#VARIABLES AND CREDENTIALS
load_dotenv(dotenv_path="credentials.env")
sender = os.getenv("sender")
recipient = os.getenv("receiver")
smtppass = os.getenv("smtp_password")
access_token = os.getenv("access_tok")
access_token_sec = os.getenv("access_tok_sec")
cons_key = os.getenv("cons_key")
cons_secret = os.getenv("cons_secret") 

# Authentication Of Tweepy. 
auth = tweepy.OAuthHandler(cons_key, cons_secret)
auth.set_access_token(access_token, access_token_sec)
api = tweepy.API(auth)

nlp = spacy.load("en_core_web_sm") #language model to be used for POS tagging
intensityanalyzer = SentimentIntensityAnalyzer() # To compute compund sentiment scores for target brand

def create_df():
    return pd.DataFrame(columns=["Date", "User", "Tweet", "LikeCount", "Retweets"])
target_entity = create_df()
competitors = create_df() 

def get_target_tweets(targetBrand, tweetCount):
    i = 0
    for tweet in tweepy.Cursor(api.search_tweets,q=targetBrand,count=100,lang="en", excludes="retweets").items():
        target_entity.loc[i, "Date"] = tweet.created_at
        target_entity.loc[i, "User"] = tweet.user.name
        target_entity.loc[i, "Tweet"] = tweet.text
        target_entity.loc[i, "LikesCount"] = tweet.favorite_count
        target_entity.loc[i, "RetweetsCount"] = tweet.retweet_count
        i = i + 1
        if i >= tweetCount:
            break 
        else:
            pass
def get_competitors_tweets(competing_brands, tweetCount):
    i = 0
    for tweet in tweepy.Cursor(api.search_tweets, q=competing_brands, count=100, lang="en", exclude="retweets").items():
        competitors.loc[i, "Date"] = tweet.created_at
        competitors.loc[i, "User"] = tweet.user.name 
        competitors.loc[i, "Tweet"] = tweet.text 
        competitors.loc[i, "LikesCount"] = tweet.favorite_count
        competitors.loc[i, "RetweetsCount"] = tweet.retweet_count 
        i = i + 1
        if i >= tweetCount:
            break 
        else:
            pass
#function to clean the data for machine learning. 
def clean_tweets(tweets):
    stopwords = nltk.corpus.stopwords.words("english")
    newStopWords = ("kwanza", "hello", "hi", "kwa", "hey",
                        "ni", "si", "na", "tu", "za", "yake", "mm", "us", "dm", "rm", "dw", "woukd", "like", "cw", "still", "see", "caro",
                        "sisi", "nyinyi", "watu", "wa", "guys", "za", "zetu", "yetu", "yako", "yangu", "zangu", "wetu", "si", "ni", "na",
                        "use", "also", "id", "rt","also", "tried", "inaisha", "sa", "got", "izo", "wako", "btw", "alone",
                        "kabisaaaaaa", "walisema", "sio", "via", "amp", "ata", "bw", "kinda", "ok", "every", "want", "would", "till", "need")
    stopwords.extend(newStopWords)
    #removing extra spaces
    space_pat = re.compile(r'\s+') 
    tweets = tweets.str.replace(space_pat, " ")
    #removing @ user name
    user_pat = re.compile(r'@[\w\-]+')    
    tweets = tweets.str.replace(user_pat, "") 
    #removing url links
    urls_pat = re.compile(r"http\S+")
    tweets = tweets.str.replace(urls_pat, "")
    #removing numbers and punctuation
    tweets = tweets.str.replace("[^a-zA-Z]", " ")
    #converting all text to lower case
    tweets = tweets.str.lower() 
    #stopword removal  
    tweets = tweets.apply(lambda x: " ".join([word for word in x.split() if word not in (stopwords)]))
    return tweets 
#Function to clean the data for sentiment intensity analyzer sentiment scores
def clean_4_vader(tweets):
    space_pat = re.compile(r'\s+')
    tweets = tweets.str.replace(space_pat, " ")
    #user name removal 
    user_pat = re.compile(r"@[\w\-]+")
    tweets = tweets.str.replace(user_pat, "")
    #Removing URL Links from the tweets 
    urls_pattern = re.compile(r'https:\S+')
    tweets = tweets.str.replace(urls_pattern, "")
    return tweets

get_target_tweets("Safaricom", 2000)
get_competitors_tweets("AirtelKe" and "TelkomKenya", 2000)

#Cleaning raw tweets to clean text for modelling. 
target_entity['clean_tweets'] = clean_tweets(target_entity['Tweet'])
competitors['clean_tweets'] = clean_tweets(competitors['Tweet'])

#Loading Machine Learning models for classification
def load_model(model_name):
    return joblib.load(model_name)
single_entity = load_model("single_entity_pipeline.sav")
com_model = load_model("com_pipeline.sav")

#Making predictions for the target brand
target_entity["PredictedSentiments"] = single_entity.predict(target_entity['clean_tweets'])

#Competing brands predictions
com_predictions = com_model.predict(competitors['clean_tweets'])
predictions_df = pd.DataFrame(com_predictions, columns=["B1_Pos", "B1_Neut", "B1_Neg", 
                                                       "B2_Pos", "B2_Neut", "B2_Neg"])
competitors_df = pd.concat([competitors, predictions_df], axis=1) 

#Overall Analysis of predicted results for target brand
def get_sentiment_score(sentiment, df):
    return len(df[df['PredictedSentiments'] == sentiment])/len(df)
pos_score = get_sentiment_score("Positive", target_entity)
neg_score = get_sentiment_score("Negative", target_entity)

#Positivity scores for competing brands
def get_competitor_scores(competitorsdf):
    b1pos = len(competitorsdf[competitorsdf['B1_Pos'] == 1]) /len(competitorsdf)
    b2pos = len(competitorsdf[competitorsdf['B2_Pos'] == 1]) / len(competitorsdf)
    return b1pos, b2pos
b1PosScore, b2PosScore = get_competitor_scores(competitors_df)

#Computing compund scores for target brand
target_entity['VaderText'] = clean_4_vader(target_entity['Tweet'])
target_entity['CompoundSentimentScore'] = target_entity['VaderText'].apply(lambda x: intensityanalyzer.polarity_scores(str(x))['compound'])

#Getting the most polarized tweets from target entity
def get_polarized_tweets(df):
    pos_tweet = df[df['CompoundSentimentScore'] == df['CompoundSentimentScore'].max()]['Tweet'].values
    neg_tweet = df[df['CompoundSentimentScore'] == df['CompoundSentimentScore'].min()]['Tweet'].values
    return pos_tweet, neg_tweet
mostPositive, mostNegative = get_polarized_tweets(target_entity )

#Getting the most viral tweet based on retweets/likes
def get_viral_tweet():
    '''
    We extract viral tweets based on a given  number of Likes/Retweets count. 
    User will choose which to use
    '''
    viralTweetsDf = target_entity[target_entity['LikesCount'] == target_entity['LikesCount'].max()]
    tweet = viralTweetsDf['Tweet'].values  
    cpd_score = viralTweetsDf['CompoundSentimentScore'].values  
    return tweet, cpd_score 
viral_tweet, viral_tweet_score = get_viral_tweet()
#BRAND ASPECT EXTRACTION:
#Viral tweet aspect extraction
def get_viral_aspects():
    viral_aspects = {}
    docs = nlp(str(viral_tweet))
    for chunk in docs.noun_chunks:
        adj = [] 
        noun = "" 
        for tok in chunk:
            if tok.pos_ == "NOUN":
                noun = tok.text 
            if tok.pos_ == "ADJ":
                adj.append(tok.text) 
        if noun:
            viral_aspects.update({noun:adj})
    return viral_aspects
viralTweetAspects = get_viral_aspects()

#WE  EXTRACT POSITIVE AND NEGATIVE BRAND ASPECTS FOR THE TARGET ENTITY. 
def get_brand_aspects(aspect_type):
    posTweets = target_entity[target_entity['CompoundSentimentScore'] > 0.85].drop_duplicates(subset="Tweet")
    negTweets = target_entity[target_entity['CompoundSentimentScore'] <- 0.75].drop_duplicates(subset="Tweet")
    brand_aspects = {}
    viral_aspects = {}
    if aspect_type == "positive":
        docs = nlp(str(posTweets['VaderText']))
        for chunk in docs.noun_chunks:
            adj = []
            noun = ""
            for tok in chunk:
                if tok.pos_ == "NOUN":
                    noun = tok.text 
                if tok.pos_ == "ADJ":
                    adj.append(tok.text)
            if noun:
                brand_aspects.update({noun:adj})
            
    elif aspect_type == "negative":
        docs = nlp(str(negTweets['VaderText']))
        for chunk in docs.noun_chunks:
            adj = []
            noun = ""
            for tok in chunk:
                if tok.pos_ == "NOUN":
                    noun = tok.text 
                if tok.pos_ == "ADJ":
                    adj.append(tok.text)
            if noun:
                brand_aspects.update({noun:adj})
        

    
    return brand_aspects 
posAspects, negAspects = get_brand_aspects("positive"), get_brand_aspects("negative")
#CREATING A DATAFRAME TO STORE THE RESULTS
def store_results():
    targetEntityResults=  {
                            "Date": [datetime.today().strftime("%Y-%m-%d")],
                            "BrandPositiity": [pos_score],
                            "BrandNegativity":[neg_score], 
                            "PositiveTweet":[mostPositive], 
                            "NegativeTweet":[mostNegative], 
                            "PosBrandAspects":[str(posAspects)],
                            "NegBrandAspects":[str(negAspects)],
                            "ViralTweet":[viral_tweet],
                            "ViralTweetScore":[viral_tweet_score], 
                            "ViralAspects":[str(viralTweetAspects)]


    }
    tgt_cols = []
    for tgtkey in targetEntityResults.keys():
        tgt_cols.append(tgtkey)
    comparison_results = {
                            "Date": [datetime.today().strftime("%Y-%m-%d")],
                            "SafaricomPositivity": [pos_score], 
                            "AirtelPositivity": [b1PosScore],
                            "SafaricomPositivity":[b2PosScore]
    }
    com_cols = []
    for key in comparison_results.keys():
        com_cols.append(key)
    targetentityresultsdf = pd.DataFrame(targetEntityResults, columns=tgt_cols)
    comparisonresults = pd.DataFrame(comparison_results, columns = com_cols )
    return targetentityresultsdf, comparisonresults

target_transformed_df, comparison_transformed_df = store_results()
#SETTING UP CONNECTION WITH DATABASE TO STORE RESULTS

#SEND ALERTS
print("Successful run!!")

