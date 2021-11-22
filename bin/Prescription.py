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


import sys
import os
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators


@Configuration()
class Prescription(StreamingCommand):
    field1= Option(
        doc='''
        **Syntax:** ***totalImages=3
        **Description:** Restricts image search''',
        require=False)

    def stream(self, records):
        for record in records:
            origText = record['text']
            outputFile = open("/tmp/1.txt", "w")
            outputFile.write(origText)
            outputFile.close()
            

            # Medication names and codes as JSON dictionary.
            codesdict = {
              "paracetamol": 1,
              "panadol": 1,
              "aspirin": 2,
              "penicillin":3  
            }

            #Data preparation. Creating input file.
            outputFile = open("/tmp/input.txt", "w")
            outputFile.write(origText)
            outputFile.close()

            #Reading input file. 
            inputFile = open('/tmp/input.txt', 'r')
            Lines = inputFile.readlines()
            ouputList = []
            def generatePrescription(Lines): 
                for line in Lines:
                    sublist =[]
                    code = ''
                    dosage =''
                    if not line.strip():
                        break #Anything below a line which contains only spaces should be ignored.
                    line = ' '.join(line.split()) #Convert multiple spaces into single space in each line
                    lineSplit = line.split(" ") #Split each line into medication name and dosage instructions 
                    firstElement = lineSplit[0]
                    restElements = lineSplit[1:]
                    for item in restElements:
                        dosage  =dosage + " "+ item
                    try:
                        code = codesdict[firstElement.lower()]  #Convert medication into lowercase for case 
                        #insensitive dictionary lookup
                    except Exception:
                        continue #If a code is not found ignore and continue
                    sublist.append(code)
                    sublist.append(dosage)
                    ouputList.append(sublist)
                    return ouputList
                
            generatePrescription(Lines)
            
            
            
            yield{'Text': ouputList}


            
dispatch(Prescription, sys.argv, sys.stdin, sys.stdout, __name__)


