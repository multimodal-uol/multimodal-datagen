import json
import sys,time, os
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append('/opt/anaconda3/pkgs/python-2.7.13-heccc3f1_16/lib/python2.7/lib-dynload')
#import pandas as pd
from datetime import datetime
from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
import traceback


class CaptionsObj:  
    def __init__(self, timestamp, text, image, search_id, image_search_id, current_time,img_urls):
        self.AAA = current_time  #required, otherwise similar JSONs are not indexed. 
        self.image = image
        self.timestamp = timestamp
        self.text = text
        self.search_id = search_id
        self.image_search_id = image_search_id
        self.current_time = current_time
        self.img_urls = img_urls

def detectImageDemographics(testing, modelType, tweetJSON, search_id, image_search_id, totalImages, addImageDescription, confidenceScore):
    if totalImages == 'all':
        totalImages = 3000
    confidenceScore = float(confidenceScore)
    model = ''
    if testing =='True' :
        jsonTweetsDirectory = '/usr/testmedia/'
    else:
        jsonTweetsDirectory ='/opt/twitterdata/imagedata'
    outputpath = jsonTweetsDirectory+'/'+image_search_id+'.json'

    jsonParentDirectory ='/opt/twitterdata/'
    jsonDirecory=jsonParentDirectory+'images/'+image_search_id+'/'
    #os.mkdir(jsonDirecory)
    
    model = ''
    if modelType == 'NSFW':
        model = 'e9576d86d2004ed1a38ba0cf39ecb4b1'
    elif modelType == 'Apparel':
        model = '72c523807f93e18b431676fb9a58e6ad'
    elif modelType == 'Celebrity':
        model = 'cfbb105cb8f54907bb8d553d68d9fe20'
    elif modelType == 'Face':
        model = 'f76196b43bbd45c99b4f3cd8e8b40a8a'
    elif modelType == 'Logo':
        model = 'c443119bf2ed4da98487520d01a0b1e3' 
    elif modelType == 'Color':
        model = 'eeed0b6733a644cea07cf4c60f87ebb7'
        
    k=0
    listCaption = []
    concepts = ''
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
                        if modelType== 'Demographics' or modelType== 'Travel' or modelType== 'Food' or modelType== 'General': 
                            try:
                                stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())
                                metadata = (('authorization', 'Key ed69d46b58a24e96aa7184ed2f635931'),)
                                post_workflow_results_response = stub.PostWorkflowResults(
                                    service_pb2.PostWorkflowResultsRequest(
                                        workflow_id=modelType,
                                        inputs=[
                                            resources_pb2.Input(
                                                data=resources_pb2.Data(
                                                    image=resources_pb2.Image(
                                                        url=i
                                                    )
                                                )
                                            )
                                        ]
                                    ),
                                    metadata=metadata
                                )
                                if post_workflow_results_response.status.code != status_code_pb2.SUCCESS:
                                    raise Exception("Post workflow results failed, status: " + 
                                                    post_workflow_results_response.status.description)

                                results = post_workflow_results_response.results[0]

                                if modelType== 'Demographics':
                                    concepts = results.outputs[2].data.regions[0].data.concepts
                                elif modelType== 'Travel' or modelType== 'Food' or modelType == 'General':
                                    concepts = results.outputs[0].data.concepts
                                        
                                for concept in concepts:
                                    if concept.value > confidenceScore:
                                        allConcepts=allConcepts+ ' '+concept.name
                            except:
                                f = open("/tmp/commands.txt", "w")
                                f.write(str(traceback.format_exc()))
                                f.close()
                                k=k-1
                                continue
                        else:
                            try:
                                channel = ClarifaiChannel.get_grpc_channel()
                                stub = service_pb2_grpc.V2Stub(channel)
                                metadata = (('authorization', 'Key ed69d46b58a24e96aa7184ed2f635931'),)
                                post_model_outputs_response = stub.PostModelOutputs(
                                    service_pb2.PostModelOutputsRequest(
                                        model_id=model,
                                        #version_id="{THE_MODEL_VERSION_ID}",  # This is optional. Defaults to the latest model version.
                                        inputs=[
                                            resources_pb2.Input(
                                                data=resources_pb2.Data(
                                                    image=resources_pb2.Image(
                                                        url=i
                                                    )
                                                )
                                            )
                                        ]
                                    ),
                                    metadata=metadata
                                )
                                if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
                                    raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)
                                output = post_model_outputs_response.outputs[0]
                                if modelType == 'Logo' or modelType== 'Apparel' or modelType== 'Face': 
                                    for region in output.data.regions:
                                        for concept in region.data.concepts:
                                            if concept.value > confidenceScore:
                                                allConcepts=allConcepts+ ' '+concept.name
                                elif modelType == 'Color':
                                    concepts = output.data.colors
                                elif modelType == 'Celebrity' or modelType== 'NSFW' :
                                    concepts = output.data.concepts
                                for concept in concepts:
                                    if concept.value > confidenceScore:
                                        if modelType == 'Color':
                                            allConcepts=allConcepts+ ' '+concept.w3c. name
                                        else:
                                            allConcepts=allConcepts+ ' '+concept.name
                            except:
                                f = open("/tmp/commands.txt", "w")
                                f.write(str(traceback.format_exc()))
                                f.close()
                                k=k-1
                                continue
                    imageFile = open("/tmp/"+image_search_id+".txt", "a")
                    imageFile.write(i +','+ allConcepts+"\n")
                    imageFile.close()
                    listCaption.append(CaptionsObj(str(timestamp_list), text, 
                                           allConcepts, search_id, image_search_id,current_time_list,i))
        with open(outputpath, 'w') as f:
                json.dump([ob.__dict__ for ob in listCaption], f,indent=4)
                
def detectVideos(testing, modelType, tweetJSON, search_id, video_search_id, totalVideos, addVideoDescription, confidenceScore):
    if totalVideos == 'all':
        totalVideos = 3000
    with open(tweetJSON) as f:
        jsonData = json.load(f)
        current_time_list = time.strftime("%Y%m%d-%H%M%S") #row['current_time']
        for row in jsonData:
            video_urls_list = row['video_urls']
            for i in video_urls_list:
                if k<int(totalVideos):
                    k=k+1
                    imageFile = open("/tmp/"+video_search_id+".txt", "a")
                    imageFile.write(i+"\n")
                    imageFile.close()

