from __future__ import absolute_import, division, print_function, unicode_literals
import app
import os,sys,subprocess
import time
from datetime import datetime
import json, csv

sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')
splunkhome = os.environ['SPLUNK_HOME']
sys.path.append(os.path.join(splunkhome, 'etc', 'apps', 'searchcommands_app', 'lib'))
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators
from splunklib import six
import urllib.request
import requests
import cv2
import numpy as np
from urllib.parse import urlparse

INTERNAL_BASE_DIR ='/opt/splunk/etc/apps/multimodal-datagen/appserver/static/Densecap_Images'
EXTERNAL_BASE_DIR = 'http://35.246.69.64:8000/en-GB/static/app/multimodal-datagen/Densecap_Images'

@Configuration()
class DenseCap(StreamingCommand):
    image_url = Option(
        require=True,
        doc='''
        **Syntax:** **image_url=***<fieldname>*
        **Description:** Name of the field that will contain the image urls''',
        validate=validators.Fieldname())
    
    source = Option(
        require=True,
        doc='''
        **Syntax:** **source=***<source>*
        **Description:** Name of the source that will contain the image urls''',
        )
    
    search_id = Option(
        require=False,
        doc='''
        **Syntax:** **search_id=***<search_id>*
        **Description:** search_id of the source that will contain the image urls''',
        )
    
    def stream(self, records):
        for record in records:
            
            url = record[self.image_url]
            
            try:
                r = requests.post(
                    "https://api.deepai.org/api/densecap",
                    data={
                        'image': url,
                    },
                    headers={'api-key': '77355c99-4b0b-48b2-aa00-fdfa089ff96b'}
                )
                api_results = r.json()
                output = api_results['output']
                #img = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1) 

            except:
                continue
                
            a = urlparse(url)
            image_name = os.path.basename(a.path)
            url_response = urllib.request.urlopen(url)
            img = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)            

            draw_img=img.copy()
            res=[]
            count = 1
            
            
            for row in api_results['output']['captions']:
                caption = row['caption']
                bounding_box = row['bounding_box']
                confidence = row['confidence']
                [x,y,w,h] = bounding_box

                if confidence > 0.8:
                    res.append(str(count)+". "+caption + " ({}).".format(round(confidence,2)))
                    color = list(np.random.random(size=3) * 256)
                    cv2.rectangle(draw_img, (x, y), (x+w, y+h), color, 6)
                    cv2.putText(draw_img, str(count), (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)                    
                    count+=1
            
            internal_image_dir = os.path.join(INTERNAL_BASE_DIR,self.source)
            external_image_dir = os.path.join(EXTERNAL_BASE_DIR,self.source)
            
            if self.source != 'url':
                internal_image_dir = os.path.join(internal_image_dir,self.search_id)
                external_image_dir = os.path.join(external_image_dir,self.search_id)
                if not os.path.exists(internal_image_dir):
                    os.makedirs(internal_image_dir)
            
            record['original image'] = url
            record['caption'] = res
            record['image with bounding box'] = os.path.join(external_image_dir,image_name)

            yield record
            cv2.imwrite(os.path.join(internal_image_dir,image_name),draw_img)
    
    
    
dispatch(DenseCap, sys.argv, sys.stdin, sys.stdout, __name__)
