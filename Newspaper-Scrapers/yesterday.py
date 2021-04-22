from newspaper import build
from newspaper import Article
from datetime import datetime, timedelta
import json
import time
import glob
import os
import traceback

import nltk
nltk.download('punkt')

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
     

def newspaper_parser (article):
    date_published = str(article.publish_date).replace(' ','T').replace(']','').replace('[','')[0:19]
    print('date_published '+date_published )
    if date_published[:10] == datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d'): 
        title = article.title,
        news_outlet = article.meta_site_name
        if not news_outlet:
            news_outlet = 'The Independent'
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
        paper = build(line)
        publisher=''
        global listArticles
        listArticles = []
        i = 0
        for article in paper.articles:
            print(article.url)
            article = Article(article.url)
            try:
                article.build()
            except Exception:
                print(traceback.format_exc())
                time.sleep(60)
                continue
            data = newspaper_parser(article)
            if i<1:
                publisher = str(article.meta_site_name)
                i=i+1
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
    
        
    



