import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

import json
import os
import tarfile
import warnings

import numpy as np
import pandas as pd
import requests
import spacy

import wget
# from elevate import elevate
from spacy.lang.en import English
import scattertext as st

from json import loads
from lxml import html
from requests import Session
from concurrent.futures import ThreadPoolExecutor as Executor
from itertools import count




#we are not using jupyter notebook, so no jsonify

# #pred logic, grabs information for us from post
# def ValuePredictor(to_predict_list):
#     to_predict = np.array(to_predict_list).reshape(1, -1)#function to capture what comes into post
#     loaded_model = pickle.load(open("termFr01.pkl", "rb"))
#     result = loaded_model.get_termfreq_from_url(to_predict_list, from_isbn=False)
#     return result[0]#similar to the logic of make_predict function
# #all the above does the prediction for you

warnings.filterwarnings('ignore')

lg_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz"

r = requests.get(lg_url, allow_redirects=True)
open('sm.zip', 'wb').write(r.content)

tar = tarfile.open('sm.zip', "r:gz")
tar.extractall('down_sm')

nlp = spacy.load("./down_sm/en_core_web_sm-2.1.0/en_core_web_sm/en_core_web_sm-2.1.0")

def ValuePredictor(yelp_url, from_isbn=False):
    '''Takes a url, scrape site for reviews
    and calculates the term frequencies 
    sorts and returns the top 10 as a json object
    containing term, highratingscore, poorratingscore.'''
    
    base_url = "https://www.yelp.com/biz/" # add business id
    api_url = "/review_feed?sort_by=date_desc&start="
    bid = "flower-child-addison-2" # business id

    class Scraper():
        def __init__(self):
            self.data = pd.DataFrame()

        def get_data(self, n, bid=bid):
            with Session() as s:
                with s.get(base_url+bid+api_url+str(n*20)) as resp: #makes an http get request to given url and returns response as json
                    r = loads(resp.content) #converts json response into a dictionary
                    _html = html.fromstring(r['review_list']) #loads from dictionary

                    dates = _html.xpath("//div[@class='review-content']/descendant::span[@class='rating-qualifier']/text()")
                    reviews = [el.text for el in _html.xpath("//div[@class='review-content']/p")]
                    ratings = _html.xpath("//div[@class='review-content']/descendant::div[@class='biz-rating__stars']/div/@title")

                    df = pd.DataFrame([dates, reviews, ratings]).T

                    self.data = pd.concat([self.data,df])

        def scrape(self): #makes it faster
            # multithreaded looping
            with Executor(max_workers=40) as e:
                list(e.map(self.get_data, range(10)))

    s = Scraper()
    s.scrape()
    df = s.data
    df = df.sample(100)
    
    nlp.Defaults.stop_words |= {'he','check-in','=','= =','male','u','want', 'u want', 'cuz','him',"i've", 'deaf','on', 'her','told','told him','ins', '1 check','I', 'i"m', 'i', ' ', 'it', "it's", 'it.','they','coffee','place','they', 'the', 'this','its', 'l','-','they','this','don"t','the ', ' the', 'it', 'i"ve', 'i"m', '!', '1','2','3','4', '5','6','7','8','9','0','/','.',','}

    corpus = st.CorpusFromPandas(df, 
                             category_col=2, 
                             text_col=1,
                             nlp=nlp).build()

    term_freq_df = corpus.get_term_freq_df()
    term_freq_df['highratingscore'] = corpus.get_scaled_f_scores('5.0 star rating')

    term_freq_df['poorratingscore'] = corpus.get_scaled_f_scores('1.0 star rating')

    df = term_freq_df.sort_values(by= 'poorratingscore', ascending = False)

    df['highratingscore'] = round(df['highratingscore'], 2)
    df['poorratingscore'] = round(df['poorratingscore'], 2)
    
    list1 = []
    for i in df.index[:10]:
        list1.append(i)

    return json.dumps(list1)

#app
app=Flask(__name__)

#routes
@app.route('/')#defaults to this just in case, legacy reasons
@app.route('/index')
def index():
    return flask.render_template('index.html')#we are going to have a form

#we have to have something to hold and run the results page
@app.route('/result', methods = ['POST'])
def result():#will capture our predictions, handles result
#result will grab post from index.html, instead of host
    if request.method == 'POST':
        #take information from form with nominal data, let html do the work convert to dictionary
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(str, to_predict_list))
        result = ValuePredictor(to_predict_list)
    return render_template("results.html", prediction=result)
#app run, so we don't export as an actual app
if __name__ == '__main__':
    app.run(port=9000, debug=True)
#windows go to system variables to shut down.