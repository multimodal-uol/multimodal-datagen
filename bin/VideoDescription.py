# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:13:04 2020

@author: Malcom
"""
import exec_anaconda
#exec_anaconda.exec_anaconda()
import sys,os
#sys.path.append(r'C:\anaconda\envs\tensorflow\Lib\site-packages')
import cv2
os.environ["LD_LIBRARY_PATH"] = "/opt/splunk/lib/:$LD_LIBRARY_PATH"
#os.system('export LD_LIBRARY_PATH="/opt/splunk/lib/:$LD_LIBRARY_PATH"')
import pafy
import numpy as np
from clarifai.rest import ClarifaiApp
from urllib.parse import urlparse, parse_qs

#url='https://www.youtube-nocookie.com/embed/AloNERbBXcc?wmode=opaque&feature=oembed'

def extract_video_id(url):
    # Examples:
    # - http://youtu.be/SA2iWivDJiE
    # - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    # - http://www.youtube.com/embed/SA2iWivDJiE
    # - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in {'www.youtube.com', 'youtube.com','www.youtube-nocookie.com'}:
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]
    # fail?
    return None

def detectVideo(imagePath):
    app = ClarifaiApp()
    model = app.public_models.general_model
    allConcepts=''

    try:
        response = model.predict_by_filename(imagePath)
    except:
        return allConcepts
    concepts = response['outputs'][0]['data']['concepts']
    for concept in concepts:
        allConcepts=allConcepts+ ' '+concept['name']
    return allConcepts

def split_into_parts(number, n_parts):
    return np.linspace(0, number, n_parts+1,dtype=int)[1:]

def extractFrames(VideoURL, n_parts):
    description=''
    try:
        newURL='http://www.youtube.com/watch?v='+extract_video_id(VideoURL)
        vPafy = pafy.new(newURL)
        play = vPafy.getbest(preftype="mp4")
        cap = cv2.VideoCapture(play.url)
        total_frames = int(cap.get(7))
        f = open("/tmp/commands.txt", "w")
        f.write(str('total_frames' +str(total_frames)))
        f.close()
        for idx in split_into_parts(total_frames, n_parts):
            cap.set(1, idx-1)
            ret, frame = cap.read()
            cv2.imwrite('temp.jpg', frame)
            description = description + detectVideo('temp.jpg')
        return description
    except:
        return description

def main(x, y):
    url = x
    n = y
    description= extractFrames(url, int(n))
    return description


