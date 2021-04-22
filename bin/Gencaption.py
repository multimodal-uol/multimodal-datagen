import json
import sys,time, os
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append('/opt/anaconda3/pkgs/python-2.7.13-heccc3f1_16/lib/python2.7/lib-dynload')
#import pandas as pd
from datetime import datetime
from clarifai.rest import ClarifaiApp


class CaptionsObj:  
    def __init__(self, timestamp, text, image, search_id, image_search_id, current_time,img_urls):  
        self.image = image
        self.timestamp = timestamp
        self.text = text
        self.search_id = search_id
        self.image_search_id = image_search_id
        self.current_time = current_time
        self.img_urls = img_urls

def detectImage(testing, modelType, tweetJSON, search_id, image_search_id, totalImages, addImageDescription, confidenceScore):
    if totalImages == 'all':
        totalImages = 3000
    confidenceScore = float(confidenceScore)
    app = ClarifaiApp()
    model = ''
    if modelType == 'General':
        model = app.public_models.general_model
    elif modelType == 'Demographics':
        model = app.models.get(model_id="c0c0ac362b03416da06ab3fa36fb58e3")
    elif modelType == 'Travel':
        model = app.models.get(model_id="eee28c313d69466f836ab83287a54ed9")
    elif modelType == 'Food':
        model = app.models.get(model_id="bd367be194cf45149e75f01d59f77ba7")
    elif modelType == 'NSFW':
        model = app.models.get(model_id="e9576d86d2004ed1a38ba0cf39ecb4b1")
    elif modelType == 'Apparel':
        model = app.models.get(model_id="e0be3b9d6a454f0493ac3a30784001ff")
    elif modelType == 'Celebrity':
        model = app.models.get(model_id="e466caa0619f444ab97497640cefc4dc")
    elif modelType == 'Face':
        model = app.models.get(model_id="a403429f2ddf4b49b307e318f00e528b")
    elif modelType == 'Logo':
        model = app.models.get(model_id="c443119bf2ed4da98487520d01a0b1e3")
    elif modelType == 'Color':
        model = app.models.get(model_id="eeed0b6733a644cea07cf4c60f87ebb7")
    if testing =='True' :
        jsonTweetsDirectory = '/usr/testmedia/'
    else:
        jsonTweetsDirectory ='/opt/twitterdata/imagedata'
    outputpath = jsonTweetsDirectory+'/'+image_search_id+'.json'

    jsonParentDirectory ='/opt/twitterdata/'
    jsonDirecory=jsonParentDirectory+'images/'+image_search_id+'/'
    #os.mkdir(jsonDirecory)
    k=0
    listCaption = []
    with open(tweetJSON) as f:
        jsonData = json.load(f)
        current_time_list = time.strftime("%Y%m%d-%H%M%S") #row['current_time']
        for row in jsonData:
            img_list = row['img_urls']
            timestamp_list = row['timestamp']
            img_urls_list = row['img_urls']
            text = row['text']
            
            for i in img_list:
                if k<int(totalImages):
                    allConcepts='' 
                    k=k+1
                    if addImageDescription =='True':
                        try:
                            response = model.predict_by_url(url=i)
                            if (modelType == 'Demographics' or modelType == 'Celebrity' or modelType == 
                                'Face' or modelType == 'Logo'):
                                concepts = response['outputs'][0]['data']['regions'][0]['data']['concepts']
                            elif (modelType == 'Color'):
                                concepts = response['outputs'][0]['data']['colors']
                            else:
                                concepts = response['outputs'][0]['data']['concepts']
                            for concept in concepts:
                                if concept['value'] > confidenceScore:
                                    if modelType == 'Color':
                                        allConcepts=allConcepts+ ' '+concept['w3c']['name']
                                    else:
                                        allConcepts=allConcepts+ ' '+concept['name']
                        except:
                            k=k-1
                            continue
                    imageFile = open("/tmp/"+image_search_id+".txt", "a")
                    imageFile.write(i +','+ allConcepts+"\n")
                    imageFile.close()
                    listCaption.append(CaptionsObj(str(timestamp_list), text, 
                                           allConcepts, search_id, image_search_id,current_time_list,img_urls_list))
        with open(outputpath, 'w') as f:
                json.dump([ob.__dict__ for ob in listCaption], f,indent=4)

