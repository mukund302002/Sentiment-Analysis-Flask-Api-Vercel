# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:39:41 2020

@author: Anustup
"""

from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)

# Load the pre-trained model and CountVectorizer
model = joblib.load('Sentiment.pkl')
df = pd.read_csv("Tweets.csv", encoding="latin-1")
df.drop(['tweet_id', 'airline_sentiment_confidence', 'negativereason','negativereason_confidence','airline','airline_sentiment_gold','name','negativereason_gold','retweet_count','tweet_coord','tweet_created','tweet_location','user_timezone'], axis=1, inplace=True)
df['label'] = df['airline_sentiment'].map({'neutral': 0, 'positive': 1, 'negative': 2})
X = df['text']
cv = CountVectorizer()
cv.fit(X)  # Fit the CountVectorizer on the existing data

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)[0]
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
