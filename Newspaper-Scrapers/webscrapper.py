from newspaper import build
from newspaper import Article
from datetime import datetime, timedelta
import json
import time
import glob
import os
import traceback
import requests
import nltk
from bs4 import BeautifulSoup, SoupStrainer
#nltk.download('punkt')

outputdir = '/opt/coviddata/text/'
#outputdir = '/tmp/'

inputfile= 'list-websites.txt'
listArticles = []
publisher=''



class articleObj:
    def __init__(self, title, date_published, news_outlet,
                 feature_img, article_link, keywords,movies, Summary, text):
        self.title = title
        self.date_published = date_published
        self.news_outlet = news_outlet
        self.feature_img = feature_img
        self.article_link = article_link
        self.keywords = keywords
        self.movies = movies
        self.Summary = Summary
        self.text = text
     

def website_parser (article, publisher, dateToday):
    movies =[]
    date_published = dateToday + 'T00:00:00'
    title = article.title,
    website = publisher
    feature_img =  article.top_image
    article_link = article.url
    keywords = article.keywords
    
    #extract videos
    r = requests.get(article.url)
    soup = BeautifulSoup(r.content,'html.parser')
    links = soup.find_all('iframe')
    for link in links:
        try:
            video_link = link['src']
            #video_link = link['video']
        except Exception:
            continue
        if video_link:
            if  not(('googletagmanager' in video_link) or ('ServiceLogin' in video_link)) :
                #print('video_link '+video_link)
                movies.append(video_link)
    summary = article.summary
    text = article.text
    anArticle= articleObj (title, date_published, website, 
                                    feature_img, article_link, keywords,
                                    movies, summary, text)
    listArticles.append(anArticle)

def write_to_json (file_name):
    with open(file_name, 'w') as f:
        if listArticles:
            json.dump([ob.__dict__ for ob in listArticles], f)

def scrape_url ():
    with open(inputfile) as f:
         lines = [line.rstrip() for line in f]
    for line in lines:
        publisher=''
        global listArticles
        listArticles = []
        i = 0
        append_url = ''
        base_url = ''
        
        articles = []
        page = requests.get(line)    
        data = page.text
        soup = BeautifulSoup(data)
        exclude = ['/a-z', '/LoginPage', '/privacy', '/support', 
                   '/contact-us','/alexa','/referrals','/browse/','/help',
                   '/sitemap','/cookies','/language-policy','/personal-data-protection',
                   '/copyright','/our-policies','/all-topics','/news-events','/publications-data',
                   '/data-tools','about-us','mailto','/terms-conditions','/privacy-notice',
                   '/working','/visas-immigration','/abroad','/tax','/housing-local-services',
                   '/environment-countryside','/employing-people','/education','/driving','/disabilities',
                   '/justice','/citizenship','/childcare-parenting','/business','/births-deaths-marriages',
                   '/benefits','/contact','/terms-conditions','/who-brochure','/footer','/nhs-sites/','/login','sitemap/',
                   '/accessibility-statement/','/our-policies/','/jobs','/libraries','/redirect-pages']

        for link in soup.find_all('a'):
            alink = link.get('href')
            if not any(ex in alink for ex in exclude):
                articles.append(alink)
        articles.append(line)
                
        if line == 'https://www.gov.uk/coronavirus' :
            base_url = 'https://www.gov.uk'
        elif line == 'https://www.nhs.uk/conditions/coronavirus-covid-19' :
            base_url = 'https://www.nhs.uk'
        elif line == 'https://www.gov.uk/government/organisations/scientific-advisory-group-for-emergencies' :
            base_url = 'https://www.gov.uk'
        elif line == 'https://www.cdc.gov/coronavirus/2019-ncov/index.html' :
            base_url = 'https://www.cdc.gov'
        elif line == 'https://www.who.int/emergencies/diseases/novel-coronavirus-2019':
            base_url = 'https://www.who.int'
        elif line == 'https://www.nhs.uk/conditions/coronavirus-covid-19' :
            base_url = 'https://www.nhs.uk'
        else :
            base_url = line
            
        for article in articles:
            if article : 
                if  not(('http://'  in article) or ('https://' in article)) :
                    article = base_url + article 
            #print('URL '+article)
            article = Article(article)
            try:
                article.build()
            except Exception:
                #print(traceback.format_exc())
                #time.sleep(60)
                continue
            data = website_parser(article, line, dateToday)
        write_to_json('/tmp/'+dateNow+'/'+line.replace('https://www','').replace('.','').replace('/','')
                      .replace(':','').replace('https','')
                      +datetime.today().strftime('%Y-%m-%d').replace(' ','T')+'.json')
        
        
dateNow = datetime.now().strftime("%H%M%S%f")
dateToday = datetime.today().strftime('%Y-%m-%d')
if  not os.path.exists('/tmp/'+dateNow):
    os.mkdir('/tmp/'+dateNow)
scrape_url()
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
    
        

