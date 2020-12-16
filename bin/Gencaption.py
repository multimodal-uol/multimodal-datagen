import json
import sys,time
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append('/opt/anaconda3/pkgs/python-2.7.13-heccc3f1_16/lib/python2.7/lib-dynload')
#import pandas as pd

from clarifai.rest import ClarifaiApp


class CaptionsObj:  
    def __init__(self, tweet_id, timestamp, image,search_id, current_time,img_urls):  
        self.tweet_id = tweet_id
        self.image = image
        self.timestamp = timestamp
        self.search_id = search_id
        self.current_time = current_time
        self.img_urls = img_urls

def detectImage(tweetJSON, search_id, totalImages, addImageDescription):
    if totalImages == 'all':
        totalImages = 3000
    
    
    app = ClarifaiApp()
    model = app.public_models.general_model

    jsonParentDirectory='/opt/twitterdata/'
    jsonTweetsDirectory='/opt/twitterdata/tweets'
    jsonDirecory=jsonParentDirectory+'images/'+search_id+'/'
    #os.mkdir(jsonDirecory)
    k=0
    listCaption = []
    with open(tweetJSON) as f:
        jsonData = json.load(f)
        current_time_list = time.strftime("%Y%m%d-%H%M%S") #row['current_time']
        for row in jsonData:
            img_list = row['img_urls']
            id_list = row['tweet_id']
            timestamp_list = row['timestamp']
            img_urls_list = row['img_urls']
            
            for i in img_list:
                if k<int(totalImages):
                    allConcepts='' 
                    k=k+1
                    if addImageDescription =='True':
                        app = ClarifaiApp()
                        model = app.public_models.general_model
                        try:
                            response = model.predict_by_url(url=i)
                            concepts = response['outputs'][0]['data']['concepts']
                            for concept in concepts:
                                allConcepts=allConcepts+ ' '+concept['name']
                        except:
                            continue
                    listCaption.append(CaptionsObj(str(id_list), str(timestamp_list), 
                                               allConcepts,search_id,current_time_list,i))
                        
            
    with open(jsonTweetsDirectory+'/'+search_id+'-img.json', 'w') as f:
    	json.dump([ob.__dict__ for ob in listCaption], f,indent=4)

