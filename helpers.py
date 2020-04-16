import csv
import os
import logging

class Helper:
    def __init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.debug('Helper object started')
            
    def getcsvfile(self,filename):
        thisdir=os.getcwd()
        datadir=os.path.join(thisdir,'fishfiles',filename)
        #if os.path.exists(datadir):
        
        with open(datadir, 'r') as f:
            datadict=[row for row in csv.DictReader(f)]
        print(f'opening {filename} with length:{len(datadict)} and type:{type(datadict)}')
        self.logger.info(f'opening {filename} with length:{len(datadict)} and type:{type(datadict)}')
        keylist=[key for key in datadict]
        self.logger.debug(f'for filestem:{os.path.split(filename)[1]} keylist:{keylist}')
        return datadict
    
    
                
        
    
    
