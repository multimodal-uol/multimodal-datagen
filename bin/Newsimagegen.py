from __future__ import absolute_import, division, print_function, unicode_literals
import app
import os,sys,subprocess
import time
import json, csv
#os.system('/opt/anaconda3/bin/python3.7 /home/zif_multimodal_gmail_com/twint/b.py')
#import twint
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')
from clarifai.rest import ClarifaiApp
splunkhome = os.environ['SPLUNK_HOME']
sys.path.append(os.path.join(splunkhome, 'etc', 'apps', 'searchcommands_app', 'lib'))
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators
from splunklib import six
import urllib.request
import requests
import glob
from datetime import datetime
import youtube_dl
from pytube import YouTube
from VideoDescription import main

    
listArticles = []  
class imageObj:
    def __init__(self, current_time, image, title, date_published, news_outlet,feature_img, article_link):
        self.current_time = current_time
        self.feature_img = feature_img
        self.image = image
        self.title = title
        self.date_published = date_published
        self.news_outlet = news_outlet
        self.article_link = article_link
        
class videoObj:
    def __init__(self,  current_time,  movie_link, movie_desription, title, date_published, news_outlet, 
                 article_link):
        self.current_time = current_time
        self.movie_link = movie_link
        self.movie_desription = movie_desription
        self.title = title
        self.date_published = date_published
        self.news_outlet = news_outlet
        self.article_link = article_link
        
        
        
def write_to_json (file_name):
    with open(file_name, 'w') as f:
        if listArticles:
                json.dump([ob.__dict__ for ob in listArticles], f,indent=4)
                

def detectImage(imageURL):
    app = ClarifaiApp()
    model = app.public_models.general_model
    allConcepts='' 
    app = ClarifaiApp()
    try:
        response = model.predict_by_url(imageURL)
    except:
        return allConcepts
    concepts = response['outputs'][0]['data']['concepts']
    for concept in concepts:
        allConcepts=allConcepts+ ' '+concept['name']
    return allConcepts

          
@Configuration()
class NewsImageGen(StreamingCommand):
    sourceindex = Option(
        doc='''
        **Syntax:** ***sourceindex=covid/newspaper
        **Description:** Detects data source''',
        require=True)
    totalVideos = Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=True)
    totalImages = Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=True)
    addImageDescription= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    addVideoDescription= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    videoParts= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    retrieveImage= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    retrieveVideo= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)
    downloadImage= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Optionally download images''',
        require=False)
    downloadVideo= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Optionally download videos''',
        require=False)
    def stream(self, records):
        if self.sourceindex=='newspaper' :
            jsonImageDir = '/opt/newspaperdata/imagedata/'
            jsonVideoDir = '/opt/newspaperdata/videodata/'
            downloadsImageDir = '/opt/newspaperdata/downloads/images/'
            downloadsVideoDir = '/opt/newspaperdata/downloads/videos/'
        elif self.sourceindex=='covid' :
            jsonImageDir = '/opt/coviddata/imagedata/'
            jsonVideoDir = '/opt/coviddata/videodata/'
            downloadsImageDir = '/opt/coviddata/downloads/images/'
            downloadsVideoDir = '/opt/coviddata/downloads/videos/'

            self.logger.debug('CountMatchesCommand: %s', self)  
        image = ''
        movie_desription = ''
        if self.totalImages == 'all':
            self.totalImages =10000
        if self.totalVideos == 'all':
            self.totalVideos =500
        self.logger.debug('CountMatchesCommand: %s', self)  # logs command line
        image = ''
        timeNow= datetime.now().strftime("%H%M%S%f")
        number_video=0
        if int(self.totalImages) ==0 and self.retrieveImage =='True':
            yield{'title':'Maximum number of imgaes should be more than 0'}
            return 
        elif int(self.totalVideos) ==0 and self.retrieveVideo =='True':
            yield{'title':'Maximum number of videos should be more than 0'}
            return 
        for record in records:
            current_time = record['current_time']
            if  not os.path.exists(downloadsImageDir+current_time+'/'):
                os.mkdir(downloadsImageDir+current_time+'/')
            if  not os.path.exists("/tmp/"+current_time+".txt"):
                os.mknod("/tmp/"+current_time+".txt")
            if  not os.path.exists('/tmp/'+current_time+'/'):
                os.mkdir('/tmp/'+current_time+'/')
            search_id = current_time
            title = record['Title']
            date_published = record['date_published'][0:19]
            news_outlet = record['news_outlet']
            feature_img = record['feature_img']
            movies = record['videos']
            article_link = record['article_link']
            if self.retrieveImage =='True':
                number_image = sum(1 for line in open("/tmp/"+current_time+".txt"))
                if (number_image < int(self.totalImages)):
                    if self.downloadImage =='True':
                        try:
                            r = requests.get(feature_img, allow_redirects=True)
                            timeNow= datetime.now().strftime("%H%M%S%f")
                            open(downloadsImageDir+current_time+'/'+timeNow+'.jpg', 'wb').write(r.content)
                        except:
                            continue
                    if self.addImageDescription =='True':
                        image= detectImage(feature_img)
                    imageFile = open("/tmp/"+current_time+".txt", "a")
                    imageFile.write(feature_img +','+ title+"\n")
                    imageFile.close()
                    anArticle= imageObj (current_time, image, title,   date_published, news_outlet, feature_img, 
                                         article_link )
                    timeNow= datetime.now().strftime("%H%M%S%f")
                    with open('/tmp/'+current_time+'/'+timeNow+news_outlet.replace(' ','')
                              .replace('https://www','').replace('.','').replace('/','').replace(':','')
                              .replace('https','')+'.json', "w") as outfile: 
                        json_string = json.dumps(anArticle, default=lambda o: o.__dict__, sort_keys=True, indent=2)
                        outfile.write(json_string)
                    #listArticles.append(anArticle)
                    yield{'title':title,'date_published':date_published, 'news_outlet':news_outlet, 
                         'feature_img':feature_img, 'image': image, 'article_link':article_link, 
                          '_time':time.time(),'current_time':current_time}
            elif self.retrieveVideo =='True':
                if (number_video< int(self.totalVideos)):
                    if movies:
                        number_video = sum(1 for line in open("/tmp/"+current_time+".txt"))
                        if number_video < int(self.totalVideos):
                            if str(type(movies)) == '<class \'str\'>':
                                videoFile = open("/tmp/"+current_time+".txt", "a")
                                videoFile.write(movies+"\n")
                                videoFile.close()
                                if self.addVideoDescription =='True':
                                    #movies = movies.replace('&', '\&')
                                    movie_desription = main (movies, self.videoParts)
                                anArticle= videoObj (current_time,  movies, movie_desription, title, 
                                                     date_published, news_outlet, 
                                                     article_link )
                                timeNow= datetime.now().strftime("%H%M%S%f")
                                if self.downloadVideo =='True':
                                    try:
                                        YouTube(movies).streams.first().download(downloadsVideoDir+current_time+'/')
                                        os.system('mv '+downloadsVideoDir+current_time+'/'+'YouTube.*'
                                                              +' '+downloadsVideoDir+current_time+'/'+timeNow+'.mp4')
                                    except:
                                        continue
                                with open('/tmp/'+current_time+'/'+timeNow+news_outlet.replace(' ','')
                                          .replace('https://www','').replace('.','').replace('/','')
                                          .replace(':','').replace('https','')+'.json', "w") as outfile: 
                                        json_string = json.dumps(anArticle, default=lambda o: o.__dict__, sort_keys=True, indent=2)
                                        outfile.write(json_string)
                                #listArticles.append(anArticle)
                                yield{'title':title,'date_published':date_published, 'news_outlet':news_outlet, 
                                      'article_link':article_link, 
                                      'current_time':current_time, 'movies':movies, 
                                      'movie_desription':movie_desription}
                            elif str(type(movies)) == '<class \'list\'>':
                                   for movie in movies:
                                        number_video = sum(1 for line in open("/tmp/"+current_time+".txt"))
                                        if number_video < int(self.totalVideos):
                                            videoFile = open("/tmp/"+current_time+".txt", "a")
                                            videoFile.write(movie+"\n")
                                            videoFile.close()
                                            if self.addVideoDescription =='True':
                                                #movie = movie.replace('&', '\&')
                                                movie_desription = main (movie, self.videoParts)
                                            anArticle= videoObj (current_time,  movie, movie_desription, title, 
                                                                 date_published, 
                                                                 news_outlet, article_link)
                                            timeNow= datetime.now().strftime("%H%M%S%f")
                                            if self.downloadVideo =='True': 
                                                try:
                                                    YouTube(movie).streams.first().download(downloadsVideoDir+current_time+'/')
                                                    os.system('mv '+downloadsVideoDir+current_time+'/'+'YouTube.*'
                                                              +' '+downloadsVideoDir+current_time+'/'+timeNow+'.mp4')
                                                except:
                                                    continue
                                            with open('/tmp/'+current_time+'/'+timeNow+news_outlet.replace(' ','')
                                                      .replace('https://www','').replace('.','').replace('/','')
                                                      .replace(':','').replace('https','')
                                                      +'.json', "w") as  outfile: 
                                                json_string = json.dumps(anArticle, default=lambda o: o.__dict__, sort_keys=True, indent=2)
                                                outfile.write(json_string)
                                            #listArticles.append(anArticle)
                                            yield{'title':title,'date_published':date_published, 
                                                  'news_outlet':news_outlet, 
                                                  'article_link':article_link, '_time':time.time()
                                                  ,'current_time':current_time, 'movies':movie, 
                                                  'movie_desription':movie_desription}
                                            
            os.system('chmod -R 777 '+'/tmp/'+current_time)
        

        
        


dispatch(NewsImageGen, sys.argv, sys.stdin, sys.stdout, __name__)
