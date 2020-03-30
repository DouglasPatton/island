import csv
import os
import logging

class Helper:
    def __init__(self):
        self.logger=logging.getlogger(__name__)
        self.logger.debug('Helper object started')
            
    def getcsvfile(self,filename):
        thisdir=os.getcwd()
        datadir=os.path.join(thisdir,'fishfiles',filename)
        #if os.path.exists(datadir):
        
        with open(datadir, 'r') as f:
            datadict=[row for row in csv.DictReader(f)]
        print(f'opening {filename} with length:{len(datadict)} and type:{type(datadict)}')
        self.logger.info(f'opening {filename} with length:{len(datadict)} and type:{type(datadict)}')
        self.logger.debug(f')
        
        keylist=[key for row in datadict for key,val in row.items()]
        
        return datadict