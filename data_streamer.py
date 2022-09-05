import pandas as pd 
import numpy as np 
import tweepy 
from tweepy import OAuthHandler 
import nltk 
nltk.download("stopwords")
import re

 # API KEYS
cons_key = "z1pFm2CMlualoHl6e4Runo9mp"
cons_secret = "x5X6hAbmEuUov6f7O8kiltwECZ2oFXlnsUlUBzEHXSAE8tVPVw"
access_tok = "1336751394070654982-69h5b6wJ14IqvLTprnTqMjeZzIS3Dq"
access_tok_sec = "IFCWcjkNpdF1cfoyiIkKAjzGa7Kf2QJBXJzZ4Q3jNN3sE"

auth= tweepy.OAuthHandler(cons_key, cons_secret)
auth.set_access_token(access_tok, access_tok_sec)
api = tweepy.API(auth)
df = pd.DataFrame(columns=["Date", "User", "Tweet", 
"LikeCount", "Retweets"])

brand = input("Enter target brand: ")
search_param = str(brand)
def get_tweets(brand, Count):
    i = 0
    for tweet in tweepy.Cursor(api.search_tweets, q=search_param, count=100, exclude="retweets").items():
        df.loc[i, "Date"] = tweet.created_at
        df.loc[i, "User"] = tweet.user.name 
        df.loc[i, "Tweet"] = tweet.text  
        i = i + 1
        if i >= Count:
            break 
        else:
            pass
get_tweets(brand, 2000)      
def clean_tweets(tweets):
    stopwords = nltk.corpus.stopwords.words("english")
    new_stop = ("tbh", "imo", "dm", "rt", 
                            "idk", "icymi")
    stopwords.extend(new_stop)
    #Extra space removal 
    space_pat = re.compile(r'\s+')
    tweets = tweets.str.replace(space_pat, " ")
    #user name removal 
    user_pat = re.compile(r"@[\w\-]+")
    tweets = tweets.str.replace(user_pat, "")
    #Removing URL Links from the tweets 
    urls_pattern = re.compile(r'https:\S+')
    tweets = tweets.str.replace(urls_pattern, "")
    #Removing stopwords from the tweets. 
    tweets = tweets.apply(lambda x: " ".join([word for word in x.split() if word not in (stopwords)]))
    return tweets 


df['clean_tweets'] = clean_tweets(df['Tweet'])
df.to_csv(str(brand)+".csv", index=False)