from flask import Flask, request, redirect, url_for
from flask import jsonify
import json
import pandas as pd 
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
import pickle
app = Flask(__name__)
app.config['SERVER_NAME'] = 'flask-server:5000'
def prepModel():
    df = pd.read_csv('new-com-data.csv')

    # col = ['Product', 'Consumer complaint narrative']
    # df = df[col]
    # df = df[pd.notnull(df['Consumer complaint narrative'])]
    # df.columns = ['Product', 'Consumer_complaint_narrative']
    # df['category_id'] = df['Product'].factorize()[0]
    # category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
    # category_to_id = dict(category_id_df.values)
    # id_to_category = dict(category_id_df[['category_id', 'Product']].values)
    col = ['product', 'narrative']
    df = df[col]
    df = df[pd.notnull(df['narrative'])]
    df.columns = ['product', 'narrative']
    df['category_id'] = df['product'].factorize()[0]
    category_id_df = df[['product', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'product']].values)
    filename = 'categorymodel.sav'

    myModel = pickle.load(open(filename,'rb'))

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.narrative).toarray()

    return myModel, tfidf, id_to_category

def makePrediction(texts):
    myModel , tfidf, id_to_category = prepModel()
    text_features = tfidf.transform(texts)
    predictions = myModel.predict(text_features)
    
    for text, predicted in zip(texts, predictions):
        a = ('"{}"'.format(text))
        b = ("Predicted as: '{}'".format(id_to_category[predicted]))
        d = a +b 

    return jsonify(d)
    #return d

@app.route("/categories")
def categories():

    
    a = ["Hi how are you? im good thank you. I need help paying my mortgage. Yes no problem. It is e15 edd. Thats right.",
            ]
    return makePrediction(a)


 

@app.route('/makepred', methods = ['POST'])
def get_query_from_react():
    data = [request.get_json()]
    print("this is backend data",data)
    pred = makePrediction(data)
    print("this is the prediction",pred)
    return pred

if __name__ == "__main__":
    app.run(debug=True)