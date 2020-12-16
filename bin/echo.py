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

class Echo(SearchCommand):

    def __init__(self, user,word,bd,ed,location, includeImage):

        # Save the parameters
       
        
        self.user = user
        self.word = word
        self.bd = bd
        self.ed = ed
        self.location=location
        self.includeImage = includeImage

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
        if str(self.word) != '' and str(self.user) != '':
             self.output_results([{'Tweet' : ""}])

        
        if str(self.word) != '' or str(self.user) != '':
            timestr = time.strftime("%Y%m%d-%H%M%S")
            outputfile=""
      
            if (self.user !=''):
                cmd1 = str('twitterscraper ' + self.user +
                           ' --user  --lang en --dump')
                if (self.location !='all'):
                    cmd1 = ('twitterscraper "' +self.user +  ' near:' +self.location
                            +  ' within:50mi  --user"  --lang en  -o /opt/twitterdata/tweets/'+ 
                            self.user+timestr+'.json') 
                p =  subprocess.check_output(cmd1, shell=True)
                outputfile='/opt/twitterdata/tweets/'+self.user+timestr
                
                result = subprocess.check_output('imagedesc "'+p+'" id',shell=True)
                f = open("/tmp/commands.txt", "w")
                f.write(str(result))
                f.close()
                
                
               

            elif (self.word !=''):
                cmd2 = ('twitterscraper ' + self.word +' -bd '+self.bd +' -ed '
            +self.ed + ' --lang en -o /opt/twitterdata/tweets/'+self.word+timestr+'.json')
                if (self.location !='all'):
                    cmd2 = ('twitterscraper "' + self.word +  ' near:' +self.location+  ' within:50mi"  --lang en ' +' -bd '+self.bd +' -ed '
                            +self.ed+ ' -o /opt/twitterdata/tweets/'+self.word+timestr+'.json')
                f = open("/tmp/commands.txt", "w")
                f.write(cmd2)
                f.close()
                os.system(cmd2)

                outputfile='/opt/twitterdata/tweets/'+self.word+timestr
                #outputfile='/opt/twitterdata/tweets/Coronavirus20200415-121930'
                    
            if  os.path.exists(outputfile+'.json'):
                with open(outputfile+'.json') as f:
                    jsonData = json.load(f)   
                    
                if (self.word !=""):
                    searchID = self.word+timestr[10:]
                else:
                     searchID = self.user+timestr[10:]
                for i in range(len(jsonData)):   
                    jsonData[i]["search_id"] = searchID #Adding seacrh_id
                    jsonData[i]["current_time"] = timestr 

                if self.includeImage:
                    imageDescCmd = 'imagedesc '+outputfile+'.json ' + searchID
                    os.system(imageDescCmd)
                    
                os.remove(outputfile+'.json')
                with open(outputfile+'.json', 'w') as f:
                    json.dump(jsonData, f)

                i=0
                for row in jsonData:
                    if len(jsonData) >=20: 
                        if i <20:
                            self.output_results([{'search_id':str(searchID),'Tweet' : row['text']}])
                            i=i+1
                    else:
                        self.output_results([{'search_id':str(searchID),'Tweet' : row['text']}])
                

                

                    
                 
            
if __name__ == '__main__':
    Echo.execute()

