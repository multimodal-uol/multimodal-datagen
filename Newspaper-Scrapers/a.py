from newspaper.newspaper import build
from newspaper.newspaper import Article
from datetime import datetime, timedelta
from pymongo import MongoClient
client = MongoClient()
import json

db = client.News_database
listArticles = []
class articleObj:
    def __init__(self, title, date_published, news_outlet,authors, 
                 feature_img, article_link, keywords,movies, summary, text):
        self.title = title
        self.date_published = date_published
        self.news_outlet = news_outlet
        self.authors = authors
        self.feature_img = feature_img
        self.article_link = article_link
        self.keywords = keywords
        self.movies = movies
        self.summary = summary
        self.text = text

def newspaper_parser (url):
    article = Article(url)
    global publisher
    article.build()
    date_published = str(article.publish_date).replace(' ','T').replace(']','').replace('[','')
    if date_published[:10]==datetime.today().strftime('%Y-%m-%d'):
        title = article.title,
        news_outlet = article.meta_site_name
        publisher = news_outlet.replace(' ','')
        authors = article.authors
        feature_img =  article.top_image
        article_link = article.canonical_link
        keywords = article.keywords,
        movies = article.movies
        summary = article.summary
        text = article.text
        anArticle= articleObj (title, date_published, news_outlet, 
                                        authors, feature_img, article_link, keywords,
                                        movies, summary, text )
        listArticles.append(anArticle)

def write_to_json (file_name):
    with open(file_name, 'w') as f:
        json.dump([ob.__dict__ for ob in listArticles], f)


with open('1.txt') as f:
     lines = [line.rstrip() for line in f]
for line in lines:
    print(line)
    paper = build(line)
    for article in paper.articles:
    	print(article.url)

