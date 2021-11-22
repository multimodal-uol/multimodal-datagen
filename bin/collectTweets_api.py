#!/usr/bin/env python
# coding=utf-8
#
# Copyright © 2011-2015 Splunk, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"): you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import time, os, traceback
import json
import codecs
from itertools import islice
import sys, time
sys.path.append('/opt/anaconda3/lib/python3.7/site-packages')
import yaml
from searchtweets import load_credentials, gen_rule_payload, ResultStream

splunkhome = os.environ['SPLUNK_HOME']
sys.path.append(os.path.join(splunkhome, 'etc', 'apps', 'searchcommands_app', 'lib'))
from splunklib.searchcommands import dispatch,  GeneratingCommand, Configuration, Option, validators
from splunklib import six

# Define details of our account 
API_KEY = 'VzihIPxv5oFrd3SkNuBuQk9o3'
API_SECRET_KEY = 'Iq7hi4K1cZnzgD3RC1miTM6rcrHMA4aeHj3OeCsI9OvFVtX5Ej'
DEV_ENVIRONMENT_LABEL = 'datacollection'
API_SCOPE = 'fullarchive'  # 'fullarchive' for full archive, '30day' for last 31 days
RESULTS_PER_CALL = 500  # 100 for sandbox, 500 for paid tiers


@Configuration()
class Echo(GeneratingCommand):
    testing = Option(
        doc='''
        **Syntax:** ***sourceindex=covid/newspaper
        **Description:** testing or production data''',
        require=True)
    user = Option(
        doc='''
        **Syntax:** ***
        **Description:**''',
        require=False)    
    word = Option(
        doc='''
        **Syntax:** ***
        **Description:**''',
        require=False)
    bd = Option(
        doc='''
        **Syntax:** ***
        **Description:**''',
        require=False)
    ed = Option(
        doc='''
        **Syntax:** ***
        **Description:**''',
        require=False)
    location = Option(
        doc='''
        **Syntax:** ***
        **Description:**''',
        require=False)
    limit = Option(
        doc='''
        **Syntax:** ***
        **Description:**''',
        require=False)
    userName = Option(
        doc='''
        **Syntax:** ***
        **Description:**''',
        require=False)
    testing = Option(
        doc='''
        **Syntax:** ***
        **Description:**''',
        require=False)
           

    def generate(self):
        config = dict(
                search_tweets_api=dict(
                    account_type='premium',
                    endpoint=f"https://api.twitter.com/1.1/tweets/search/fullarchive/datacollection.json",
                    consumer_key=API_KEY,
                    consumer_secret=API_SECRET_KEY
                )
            )
        with open('twitter_keys.yaml', 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)
        premium_search_args = load_credentials("twitter_keys.yaml",yaml_key="search_tweets_api",env_overwrite=False)
        if str(self.word) != 'null' or str(self.user) != 'null':
            timestr = time.strftime("%Y%m%d-%H%M%S")
            if (self.userName =='-anonymous'):
                self.userName = ''
                
      
            if (self.user !='null'):
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
                           

            elif (self.word !='null'):
                searchID = self.word+timestr[10:]+self.userName
                tmpoutputfile = '/tmp/'+searchID+time.strftime("%Y%m%d-%H%M%S")+'.json'
                self.word  = self.word.replace("SPACE", " ")
                
                FROM_DATE = str(self.bd) + " 00:00"
                TO_DATE = str(self.ed) + " 00:00"
                # Put together search terms and rules from earlier
                try:
                    rule = gen_rule_payload(self.word,
                                            results_per_call=RESULTS_PER_CALL,
                                            from_date=FROM_DATE,
                                            to_date=TO_DATE
                                            )

                # Stream tweets rather than download in one go
                
                    rs = ResultStream(rule_payload=rule,
                                      max_results=int(self.limit),
                                      **premium_search_args)

                # Access API and save each tweet as single line on JSON lines file
                    with open(tmpoutputfile, 'a', encoding='utf-8') as f:
                        for tweet in rs.stream():
                            json.dump(tweet, f)
                            f.write('\n')
                except Exception:
                    f = open("/tmp/commands.txt", "w")
                    f.write(str(traceback.format_exc()))
                    f.close()     

                
                if self.testing =='True':
                    outputfile='/tmp/'+searchID+time.strftime("%Y%m%d-%H%M%S")
                else:
                    outputfile='/tmp/'+searchID
                    
            if  os.path.exists(tmpoutputfile):
                yield{'Message':'Successxx'}
                
            else:
                yield{'Message':'No Results Found!'}
                return
        else:
            yield{'Message' : "Enter Input in the Textboxes "}
            return;
                    
            
                    

            
dispatch(Echo, sys.argv, sys.stdin, sys.stdout, __name__)

