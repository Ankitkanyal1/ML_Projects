import re
import os
import nltk
import joblib
import requests
import numpy as np
from bs4 import BeautifulSoup
import urllib.request as urllib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request
import time

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def clean(x):
    x = re.sub(r'[^a-zA-Z ]', ' ', x)  # replace everything that's not an alphabet with a space
    x = re.sub(r'\s+', ' ', x)  # replace multiple spaces with one space
    x = re.sub(r'READ MORE', '', x)  # remove READ MORE
    x = x.lower()
    x = x.split()
    y = []
    for i in x:
        if len(i) >= 3:
            if i == 'osm':
                y.append('awesome')
            elif i == 'nyc':
                y.append('nice')
            elif i == 'thanku':
                y.append('thanks')
            elif i == 'superb':
                y.append('super')
            else:
                y.append(i)
    return ' '.join(y)


def extract_all_reviews(url, clean_reviews, org_reviews, customernames, commentheads, ratings, sentiments):
    with urllib.urlopen(url) as u:
        page = u.read()
        page_html = BeautifulSoup(page, "html.parser")
    reviews = page_html.find_all('div', {'class': 't-ZTKy'})
    commentheads_ = page_html.find_all('p', {'class': '_2-N8zT'})
    customernames_ = page_html.find_all('p', {'class': '_2sc7ZR _2V5EHH'})
    ratings_ = page_html.find_all('div', {'class': ['_3LWZlK _1BLPMq', '_3LWZlK _32lA32 _1BLPMq', '_3LWZlK _1rdVr6 _1BLPMq']})

    for review in reviews:
        x = review.get_text()
        org_reviews.append(re.sub(r'READ MORE', '', x))
        clean_reviews.append(clean(x))

    for cn in customernames_:
        customernames.append('~' + cn.get_text())

    for ch in commentheads_:
        commentheads.append(ch.get_text())

    ra = []
    for r in ratings_:
        try:
            if int(r.get_text()) in [1, 2, 3, 4, 5]:
                ra.append(int(r.get_text()))
                sentiments.append('POSITIVE' if int(r.get_text()) > 3 else 'NEGATIVE')
            else:
                ra.append(0)
                sentiments.append('NEUTRAL')
        except:
            ra.append(r.get_text())
            sentiments.append('NEUTRAL')

    ratings += ra
    y_test = [1 if sentiment == 'POSITIVE' else 0 for sentiment in sentiments]
    return y_test

def train_models(clean_reviews, sentiments):
    vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english')
    X = vectorizer.fit_transform(clean_reviews)

    X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)

    # Train Support Vector Machine
    svm_classifier = SVC(kernel='linear', C=1, random_state=42)
    svm_classifier.fit(X_train, y_train)
    svm_predictions = svm_classifier.predict(X_test)

    return rf_classifier, rf_predictions, svm_classifier, svm_predictions, y_test

def show_overall_accuracy(clean_reviews, sentiments, rf_classifier, rf_predictions, svm_classifier, svm_predictions, y_test):
    # Combining predictions from both models
    combined_predictions = []
    for rf_pred, svm_pred in zip(rf_predictions, svm_predictions):
        # If both models predict the same sentiment or one model predicts NEUTRAL, use that sentiment
        if rf_pred == svm_pred or rf_pred == 'NEUTRAL' or svm_pred == 'NEUTRAL':
            combined_predictions.append(rf_pred)
        else:
            # If both models predict different sentiments, choose POSITIVE (you can customize this logic)
            combined_predictions.append('POSITIVE')

    overall_accuracy = accuracy_score(y_test, combined_predictions)

    print(f"Combined Model Accuracy: {overall_accuracy}")

    # Display classification report
    print("Combined Model Classification Report:")
    print(classification_report(y_test, combined_predictions))


def purchase_recommendation(combined_predictions):
    # Customize this function based on your criteria for recommending a purchase

    # For example, if the majority of predictions are POSITIVE, recommend purchase
    positive_count = combined_predictions.count('POSITIVE')
    negative_count = combined_predictions.count('NEGATIVE')
    neutral_count = combined_predictions.count('NEUTRAL')

    # If there are more POSITIVE reviews
    if positive_count > negative_count and positive_count > neutral_count:
        return "Based on reviews, we recommend purchasing the product."

    # If there are more NEGATIVE reviews
    elif negative_count > positive_count and negative_count > neutral_count:
        return "Based on reviews, consider being cautious before purchasing the product due to negative feedback."

    # If there is a tie or majority are NEUTRAL
    else:
        return "Based on reviews, there's no clear recommendation. Consider other factors before making a decision."


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results', methods=['GET'])
def result():
    url = request.args.get('url')

    nreviews = int(request.args.get('num'))
    clean_reviews = []
    org_reviews = []
    customernames = []
    commentheads = []
    ratings = []
    sentiments = []
    with urllib.urlopen(url) as u:
        page = u.read()
        page_html = BeautifulSoup(page, "html.parser")

    proname = page_html.find_all('span', {'class': 'B_NuCI'})[0].get_text()
    price = page_html.find_all('div', {'class': '_30jeq3 _16Jk6d'})[0].get_text()

    # getting the link of see all reviews button
    all_reviews_url = page_html.find_all('div', {'class': 'col JOpGWq'})[0]
    all_reviews_url = all_reviews_url.find_all('a')[-1]
    all_reviews_url = 'https://www.flipkart.com' + all_reviews_url.get('href')
    url2 = all_reviews_url + '&page=1'

    # start reading reviews and go to the next page after all reviews are read
    while True:
        x = len(clean_reviews)
        # extracting the reviews
        y_test = extract_all_reviews(url2, clean_reviews, org_reviews, customernames, commentheads, ratings, sentiments)
        url2 = url2[:-1] + str(int(url2[-1]) + 1)
        if x == len(clean_reviews) or len(clean_reviews) >= nreviews:
            break

    org_reviews = org_reviews[:nreviews]
    clean_reviews = clean_reviews[:nreviews]
    customernames = customernames[:nreviews]
    commentheads = commentheads[:nreviews]
    ratings = ratings[:nreviews]

    rf_classifier, rf_predictions, svm_classifier, svm_predictions, y_test = train_models(clean_reviews, sentiments)
    show_overall_accuracy(clean_reviews, sentiments, rf_classifier, rf_predictions, svm_classifier, svm_predictions, y_test)

     # Combining predictions from both models
    combined_predictions = []
    for rf_pred, svm_pred in zip(rf_predictions, svm_predictions):
        # If both models predict the same sentiment or one model predicts NEUTRAL, use that sentiment
        if rf_pred == svm_pred or rf_pred == 'NEUTRAL' or svm_pred == 'NEUTRAL':
            combined_predictions.append(rf_pred)
        else:
            # If both models predict different sentiments, choose POSITIVE (you can customize this logic)
            combined_predictions.append('POSITIVE')

    overall_accuracy = accuracy_score(y_test, combined_predictions)

    # Display classification report
    print("Combined Model Classification Report:")
    print(classification_report(y_test, combined_predictions))

    # Get the purchase recommendation
    recommendation = purchase_recommendation(combined_predictions)
    print(recommendation)
    # building our wordcloud and saving it
    for_wc = ' '.join(clean_reviews)
    wcstops = set(STOPWORDS)
    wc = WordCloud(width=1400, height=800, stopwords=wcstops, background_color='white').generate(for_wc)
    plt.figure(figsize=(20, 10), facecolor='k', edgecolor='k')
    plt.imshow(wc, interpolation='bicubic')
    plt.axis('off')
    plt.tight_layout()
    CleanCache(directory='static/images')
    plt.savefig('static/images/woc.png')
    plt.close()

    d = []
    for i in range(len(org_reviews)):
        x = {}
        x['review'] = org_reviews[i]
        x['cn'] = customernames[i]
        x['ch'] = commentheads[i]
        x['stars'] = ratings[i]
        x['sent'] = 'NEGATIVE' if (x['stars'] != 0 and x['stars'] in [1, 2]) else 'POSITIVE'
        d.append(x)

   
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"SVM Accuracy: {svm_accuracy}")

    # Display classification report
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))

    print("SVM Classification Report:")
    print(classification_report(y_test, svm_predictions))

    np, nn = 0, 0
    for i in d:
        if i['stars'] != 0:
            if i['stars'] in [1, 2]:
                i['sent'] = 'NEGATIVE'
                nn += 1
            else:
                i['sent'] = 'POSITIVE'
                np += 1

    return render_template('result.html', dic=d, n=len(clean_reviews), nn=nn, np=np, proname=proname, price=price,recommendation=recommendation)

@app.route('/wc')
def wc():
    return render_template('wc.html')


class CleanCache:
    '''
    this class is responsible to clear any residual csv and image files
    present due to the past searches made.
    '''

    def __init__(self, directory=None):
        self.clean_path = directory
        # only proceed if directory is not empty
        if os.listdir(self.clean_path) != list():
            # iterate over the files and remove each file
            files = os.listdir(self.clean_path)
            for fileName in files:
                print(fileName)
                os.remove(os.path.join(self.clean_path, fileName))
        print("cleaned!")


if __name__ == '__main__':
    app.run(debug=True)
