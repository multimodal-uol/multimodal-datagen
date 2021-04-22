from newspaper import build
from newspaper import Article
from datetime import datetime, timedelta
import json
import time
import glob
import os
import traceback

import nltk
#nltk.download('punkt')

outputdir = '/opt/newspaperdata/news/'
#outputdir = '/tmp/'

inputfile= 'list-newspaper.txt'
listArticles = []
publisher=''



class articleObj:
    def __init__(self, title, date_published, news_outlet,authors, 
                 feature_img, article_link, keywords,movies, Summary, text):
        self.title = title
        self.date_published = date_published
        self.news_outlet = news_outlet
        self.authors = authors
        self.feature_img = feature_img
        self.article_link = article_link
        self.keywords = keywords
        self.movies = movies
        self.Summary = Summary
        self.text = text
     

def newspaper_parser (article, publisher):
    date_published = str(article.publish_date).replace(' ','T').replace(']','').replace('[','')[0:19]
    #print('date_published '+date_published )
    if date_published[:10] == datetime.today().strftime('%Y-%m-%d'):
        if publisher == 'The Sun':
            news_outlet = article.meta_site_name
        else:
            news_outlet = publisher
        title = article.title,
        authors = article.authors
        feature_img =  article.top_image
        article_link = article.canonical_link
        keywords = article.keywords
        movies = article.movies
        summary = article.summary
        text = article.text
        anArticle= articleObj (title, date_published, news_outlet, 
                                        authors, feature_img, article_link, keywords,
                                        movies, summary, text)
        listArticles.append(anArticle)

def write_to_json (file_name):
    with open(file_name, 'w') as f:
        if listArticles:
        	json.dump([ob.__dict__ for ob in listArticles], f)

def scrape_news ():
    with open('list-newspaper.txt') as f:
         lines = [line.rstrip() for line in f]
    for line in lines:
        if line == 'https://www.theguardian.com/uk' :
            publisher = 'the Guardian'
        elif line == 'https://www.independent.co.uk' :
            publisher = 'The Independent'
        elif line == 'https://www.standard.co.uk' :
            publisher = 'Evening Standard'
        elif line == 'https://metro.co.uk' :
            publisher = 'Metro'
        elif line == 'https://www.thesun.co.uk':
            publisher = 'The Sun'
        paper = build(line)
        global listArticles
        listArticles = []
        i = 0
        for article in paper.articles:
            #print(article.url)
            article = Article(article.url)
            try:
                article.build()
            except Exception:
                #print(traceback.format_exc())
                time.sleep(60)
                continue
            data = newspaper_parser(article, publisher)
        write_to_json('/tmp/'+dateNow+'/'+publisher.replace(' ','')+str(datetime.today().strftime('%Y-%m-%d')).replace(' ','T')+'.json')
        
        
dateNow = datetime.now().strftime("%H%M%S%f")
dateToday = datetime.today().strftime('%Y-%m-%d')
if  not os.path.exists('/tmp/'+dateNow):
    os.mkdir('/tmp/'+dateNow)
scrape_news()
glob_data = []
for file in glob.glob('/tmp/'+dateNow+'/'+'*.json'):
    if os.path.getsize(file) > 0 :
        with open(file) as json_file:
            data = json.load(json_file)
            i = 0
            while i < len(data):
                glob_data.append(data[i])
                i += 1              
with open(outputdir+dateToday+'.json', 'w') as f:
    json.dump(glob_data, f, indent=4)
