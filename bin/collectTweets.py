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
import codecs
import sys

class Echo(SearchCommand):

    def __init__(self, testing, user, word, bd, ed, location, userName, limit):

        # Save the parameters
       
        
        self.user = user
        self.word = word
        self.bd = bd
        self.ed = ed
        self.location=location
        self.limit=limit
        self.userName='-'+userName
        self.testing = testing

        # Initialize the class

        SearchCommand.__init__(self, run_in_preview=True,
                               logger_name='echo_search_command')

    def handle_results(
        self,
        results,
        session_key,
        in_preview,
        ):
        cmd1=""
        cmd2=""
        data=""
        tmpoutputfile = ''
        outputfile =''
   
        if str(self.word) != '' or str(self.user) != '':
            timestr = time.strftime("%Y%m%d-%H%M%S")
            if (self.userName =='-anonymous'):
                self.userName = ''
                
      
            if (self.user !=''):
                searchID = self.user+timestr[10:]+self.userName
                tmpoutputfile = '/tmp/'+searchID+'.json' 
                cmd1 = ('/opt/anaconda3/bin/python3.7' 
                        +' /opt/splunk/etc/apps/multimodal-datagen/bin/twittercrawler.py  '
                        + self.bd +' '+self.ed + ' '+ tmpoutputfile + ' '+'null'
                        + ' '+self.location + ' '+ self.user + ' '+ self.limit)
                f = open("/tmp/commands.txt", "w")
                f.write(str(cmd1))
                f.close()     
                os.system(cmd1)
                #x= subprocess.check_output(cmd1, shell=True)
                
                if self.testing =='True':
                    outputfile='/tmp/'+searchID
                else:
                    outputfile='/opt/twitterdata/tweets/'+searchID
                           

            elif (self.word !=''):
                searchID = self.word+timestr[10:]+self.userName
                tmpoutputfile = '/tmp/'+searchID+time.strftime("%Y%m%d-%H%M%S")+'.json'
                self.word  = self.word.replace("AND", " ")
                cmd2 = ('/opt/anaconda3/bin/python3.7' 
                        +' /opt/splunk/etc/apps/multimodal-datagen/bin/twittercrawler.py  '
                        + self.bd +' '+self.ed + ' '+ tmpoutputfile + ' '+'"'+self.word +'"'
                        + ' '+self.location+' '+'null'+ ' '+ self.limit)
                
                f = open("/tmp/commands.txt", "w")
                f.write(str(cmd2))
                f.close()           
                os.system(cmd2)
                
                if self.testing =='True':
                    outputfile='/tmp/'+searchID+time.strftime("%Y%m%d-%H%M%S")
                else:
                    outputfile='/opt/twitterdata/tweets/'+searchID
                    
            if  os.path.exists(tmpoutputfile):
                jsonList = []
                with open(tmpoutputfile) as f:
                    for jsonObj in f:
                        jsonDict = json.loads(jsonObj)
                        jsonDict["search_id"] = searchID #Adding seacrh_id
                        jsonDict["current_time"] = timestr 
                        jsonList.append(jsonDict)
                    
                os.remove(tmpoutputfile)
                with codecs.open(outputfile+'.json', 'wb', encoding='utf8') as f:
                    json.dump(jsonList, f,indent=4)
                    
                if len(jsonList) ==0 :
                    self.output_results([{'Tweet':'No Results Found'}])
                    return
                
                i=0
                for row in jsonList:
                    if len(jsonList) >1000: 
                        if i <1000:
                            self.output_results([{'search_id':str(searchID),'Tweet' : row['text']}])
                            i=i+1
                    else:
                        self.output_results([{'search_id':str(searchID),'Tweet' : row['text']}])
            else:
                self.output_results([{'Message':'No Results Found!'}])
                return
        else:
            self.output_results([{'Message' : "Enter Input in the Textboxes "}])
            return;
        
            
if __name__ == '__main__':
    Echo.execute()

