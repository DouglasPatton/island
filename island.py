import numpy as np
import csv
import os

from helpers import Helper


class IslandData():
    def __init__(self):
        
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'multicluster.log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(os.path.join(logdir,handlername), maxBytes=10**6, backupCount=20)],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        
        self.datadir=os.path.join(os.getcwd(),'data')
        self.helper=Helper()
        self.pre_sandy=None
        self.post_sandy=None
        
    def getdata_firsttime(self,topickle=0):
        pre_sandy_path=os.path.join(self.datadir,'PreSandy.csv')
        post_sandy_path=os.path.join(self.datadir,'PostSandy.csv')
        self.pre_sandy=self.helper.getcsvfile(pre_sandy_path)
        self.post_sandy=self.helper.getcsvfile(post_sandy_path)
        
        

