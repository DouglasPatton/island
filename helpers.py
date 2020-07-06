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
    
    
    def getname(self,filename):
        exists=1
        while exists==1:
            if os.path.exists(filename):
                countstr=''
                dotpos=[]#dot position for identifying extensions and 
                for i,char in enumerate(filename):
                    if char=='.':
                        dotpos.append(i)
                        break
                try: 
                    lastdot=dotpos.pop(-1)
                    prefix=filename[:lastdot]
                    suffix=filename[lastdot:]
                except:
                    prefix=filename
                    suffix=''
                _pos=[]
                for i,char in enumerate(prefix):
                    if char=='_':
                        _pos.append(i)
                
                '''try:
                    last_=_pos[-1]
                    firstprefix=prefix[:last_]
                    lastprefix=prefix[last_:]'''
                if len(_pos)>0:    
                    countstr=prefix[_pos[-1]+1:]#slice from the after the last underscore to the end
                    count=1
                    if not countstr.isdigit():
                        countstr='_0'
                        count=0
                else:
                    count=0
                    countstr='_0'
                if count==1:
                    prefix=prefix[:-len(countstr)]+str(int(countstr)+1)
                else:
                    prefix=prefix+countstr
                filename=prefix+suffix
            else:
                exists=0
        return filename
        
    
    
