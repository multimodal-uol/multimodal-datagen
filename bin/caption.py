import numpy as np
import os
from PIL import Image
from pickle import load
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from utils.model import CNNModel, generate_caption_beam_search
import os
import json
import pandas as pd
import os
#import urllib.request
import requests
import sys
import datetime

from config import config

"""
    *Some simple checking
"""
imageFilename=""
tweetJSON=''
search_id=''

assert type(config['max_length']) is int, 'Please provide an integer value for `max_length` parameter in config.py file'
assert type(config['beam_search_k']) is int, 'Please provide an integer value for `beam_search_k` parameter in config.py file'

# Extract features from each image in the directory
class CaptionsObj:  
    def __init__(self, tweet_id, timestamp, image,search_id):  
        self.tweet_id = tweet_id
        self.image = image
        self.timestamp = timestamp
        self.search_id = search_id
def takeParams(arg1, arg2):
    print('take***************************')
    print(arg1+arg2)
    tweetJSON = arg1
    search_id = arg2

def downloadImage(tweetJSON):
    print('download***************************')
    df = pd.read_json(tweetJSON)

    img_list = df.get("img_urls")
    id_list = df.get("tweet_id")
    timestamp_list = df.get("timestamp")
    jsonParentDirectory='/opt/twitterdata/'
    jsonTweetsDirectory='/opt/twitterdata/tweets'
    jsonDirecory=jsonParentDirectory+'images/'+search_id+'/'
    #os.mkdir(jsonDirecory)
    k=-1
    for i in img_list:
        k=k+1
        m=0
        for j in i:
            imageFilename = str(id_list.get(k))+\
            str(timestamp_list.get(k)).replace(' ', 'T')+'_'+str(m)+'.jpg'
            #r = requests.get(j, allow_redirects=True)
            #open(jsonDirecory+imageFilename, 'wb').write(r.content)
            m=m+1

def extractFeatures(filename, model, model_type):
    print('extract*********************')
    if model_type == 'inceptionv3':
        from keras.applications.inception_v3 import preprocess_input
        target_size = (299, 299)
    elif model_type == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
        target_size = (224, 224)
	# Loading and resizing image
    image = load_img(filename, target_size=target_size)
	# Convert the image pixels to a numpy array
    image = img_to_array(image)
	# Reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# Prepare the image for the CNN Model model
    image = preprocess_input(image)
	# Pass image into model to get encoded features
    features = model.predict(image, verbose=0)
    return features

# Load the tokenizer
def detectImage(search_id):
    print('IN detect************')
    tokenizer_path = config['tokenizer_path']
    tokenizer = load(open(tokenizer_path, 'rb'))

    # Max sequence length (from training)
    max_length = config['max_length']

    # Load the model
    caption_model = load_model(config['model_load_path'])

    image_model = CNNModel(config['model_type'])
    listCaption = []  

    # Load and prepare the image
    for image_file in os.listdir(config['test_data_path']+search_id+"/"):
        if(image_file.split('--')[0]=='output'):
            continue
        if(image_file.split('.')[1]=='jpg' or image_file.split('.')[1]=='jpeg'):
            #print('Generating caption for {}'.format(image_file))
            # Encode image using CNN Model
            image = extractFeatures(config['test_data_path']+search_id+'/'+image_file, image_model, config['model_type'])
            # Generate caption using Decoder RNN Model + BEAM search
            generated_caption = generate_caption_beam_search(caption_model, tokenizer, image, max_length, beam_index=config['beam_search_k'])
            # Remove startseq and endseq
            caption = generated_caption.split()[1].capitalize()
            for x in generated_caption.split()[2:len(generated_caption.split())-1]:
                caption = caption + ' ' + x
            print(caption)
            #f = open("/tmp/allcaptions.txt", "a")
            #f.write(caption+"\n")

            listCaption.append(CaptionsObj(imageFilename[:19], imageFilename[19:38], caption,search_id))
#json_string = json.dumps([ob.__dict__ for ob in listCaption])
#print(json_string)
#print(jsonTweetsDirectory+"/"+sys.argv[2]+'-img.json')
#with open(jsonTweetsDirectory+'/'+sys.argv[2]+'-img.json', 'w') as f:
    #json.dump([ob.__dict__ for ob in listCaption], f)
        
              
   



 





 



 






 
   



 





