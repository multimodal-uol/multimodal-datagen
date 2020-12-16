#!/usr/bin/python
# -*- coding: utf-8 -*-

from search_command_example_app.search_command import SearchCommand
import os
import time

import splunk.rest
import splunk.search
import json,csv
import codecs
from itertools import islice
import subprocess
import sys
import subprocess
from clarifaipython.Gencaption import downloadImage
#from caption import *
class Echo(SearchCommand):

    def __init__(self, search_id):

        # Save the parameters
       
        
        self.search_id = search_id
        # Initialize the class

        SearchCommand.__init__(self, run_in_preview=True,
                               logger_name='echo_search_command')

    def handle_results(
        self,
        results,
        session_key,
        in_preview,
        ):
                sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')
                jsonData = ''
                outputfile = '/opt/twitterdata/tweets/'
                
                
                downloadImage('/opt/twitterdata/tweets/gautampal194781928.json','a')
                imageDescCmd = 'imagedesc '+outputfile+self.search_id +'.json'+' ' + self.search_id 
                #os.system('/tmp/1.sh&')
                #subprocess.Popen([sys.executable, '-c', 'sh /tmp/1.sh'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                #subprocess.Popen(["sh","/tmp/1.sh"],close_fds=True)
                #pid=os.fork()
                #if pid==0: # new process
                #os.system(imageDescCmd)
                    #exit()
                #x= subprocess.check_output(imageDescCmd, shell=True)
                
                
                #downloadImage('/opt/twitterdata/tweets/bollywood42020.json')
                #detectImage('bollywood42020')
                


                #f = open("/tmp/commands.txt", "w")
                #f.write(x)
                #f.close()
                if  os.path.exists(outputfile+self.search_id +'-img.json'):
                    with open(outputfile+self.search_id +'-img.json') as f:
                        jsonData = json.load(f)
                        if len(jsonData)==0 :
                            self.output_results([{'Message':'Tweets does not have any iamges'}])
                            return
                    for row in jsonData:
                        self.output_results([{'search_id':str(self.search_id ),'Image' : row['image']}])
                else:
                    self.output_results([{'Message':'Generating images multiple times for the same search_id or invalid search_id'}])
                    

            
if __name__ == '__main__':
    Echo.execute()

