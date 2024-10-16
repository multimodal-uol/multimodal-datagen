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

    def __init__(self, user,word,bd,ed,location,userName):

        # Save the parameters
       
        
        self.user = user
        self.word = word
        self.bd = bd
        self.ed = ed
        self.location=location
        self.userName='-'+userName

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
                cmd1 = str('/opt/anaconda3/bin/twitterscraper ' + self.user +
                           ' --user  --lang en -o '+ tmpoutputfile)
                if (self.location !='all'):
                    cmd1 = ('/opt/anaconda3/bin/twitterscraper "' +self.user +  ' near:' +self.location
                            +  ' within:50mi  --user"  --lang en  -o '+ tmpoutputfile)
                os.system(cmd1)
                #x= subprocess.check_output(cmd1, shell=True)
                
                outputfile='/opt/twitterdata/tweets/'+searchID
                f = open("/tmp/commands.txt", "w")
                f.write(str(cmd1))
                f.close()
                           

            elif (self.word !=''):
                searchID = self.word+timestr[10:]+self.userName
                tmpoutputfile = '/tmp/'+searchID+'.json'
                self.word  = self.word.replace("AND", " ")
                cmd2 = ('/opt/anaconda3/bin/twitterscraper ' + '"'+self.word +'"'+' -bd '+self.bd +' -ed '
            +self.ed + ' --lang en -o '+tmpoutputfile)
                if (self.location !='all'):
                    cmd2 = ('/opt/anaconda3/bin/twitterscraper "' + self.word + 
                            ' near:' +self.location+   
                            ' within:50mi"  --lang en ' +' -bd '+self.bd +' -ed '
                            +self.ed+ ' -o '+tmpoutputfile)
                
                f = open("/tmp/commands.txt", "w")
                f.write(str(cmd2))
                f.close()
                           
                os.system(cmd2)

                outputfile='/opt/twitterdata/tweets/'+searchID
                #outputfile='/opt/twitterdata/tweets/Coronavirus20200415-121930'
                    
            if  os.path.exists(tmpoutputfile):
                with codecs.open(tmpoutputfile) as f:
                    jsonData = json.load(f)
                    for element in jsonData:
                        if 'text_html' in element:
                            del element['text_html']
                    
                for i in range(len(jsonData)):   
                    jsonData[i]["search_id"] = searchID #Adding seacrh_id
                    jsonData[i]["current_time"] = timestr 
                    
                os.remove(tmpoutputfile)
                with codecs.open(outputfile+'.json', 'wb', encoding='utf8') as f:
                    json.dump(jsonData, f,indent=4)
                    
                if len(jsonData) ==0 :
                    self.output_results([{'Tweet':'No Results Found'}])
                    return
                
                i=0
                for row in jsonData:
                    if len(jsonData) >=20: 
                        if i <20:
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

