import numpy as np
import csv
import pickle
import os
import logging
from helpers import Helper
import matplotlib.pyplot as plt
#import cpi #https://github.com/datadesk/cpi
from data_viz import DataView


class IslandData(DataView):
    def __init__(self):
        
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'island.log')
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(os.path.join(logdir,handlername), maxBytes=10**6, backupCount=20)],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        
        self.datadir=os.path.join(os.getcwd(),'data')
        self.helper=Helper()
        self.datadir=os.path.join(os.getcwd(),'data')
        self.datadictlistpath=os.path.join(self.datadir,'datadictlist.pickle')
        
        self.subplotcount=0
        self.fig=None;self.ax=None
        DataView.__init__(self)
        
        
    def getDataFromCSV(self,):
        pre_sandy_path=os.path.join(self.datadir,'PreSandy.csv')
        post_sandy_path=os.path.join(self.datadir,'PostSandy.csv')
        self.pre_sandyCSV=self.helper.getcsvfile(pre_sandy_path)
        self.post_sandyCSV=self.helper.getcsvfile(post_sandy_path)
        
    def doCSVToDict(self):
        try: self.pre_sandyCSV,self.post_sandyCSV
        except: self.getDataFromCSV()
        rawdatalist=[self.pre_sandyCSV,self.post_sandyCSV]
        datalist=[]
        for data in rawdatalist:
            datalist.append(self.unpackOrderedDict(data))
        self.datadictlist=datalist
        self.pickleDataDictList(datadictlist=self.datadictlist)
        
    def pickleDataDictList(self,datadictlist=None):
        opentype='rb'
        if datadictlist:
            opentype='wb'
        try:
            with open(self.datadictlistpath,opentype) as f:
                if datadictlist:
                    pickle.dump(datadictlist,f)
                    return
                else:
                    self.datadictlist=pickle.load(f)
        except:
            if datadictlist:
                self.logger.exception('unexpected error')
                assert False,'halt'
            else:
                return self.doCSVToDict()
        return 
    
    def doDictListToNpTS(self,datadictlist,varlist,timevar='sale_year'):
        '''
        timelist dims(obs,var)
        '''
        
        timelist=[];whichdictdict={}
        for d_idx,datadict in enumerate(datadictlist):
            for time in datadict[timevar]:
                if not time in whichdictdict:
                    whichdictdict[time]=d_idx#assumes a time can't be in both dicts
                if not time in timelist:
                    timelist.append(time)
        timecount=len(timelist)
        timelist.sort()
        self.timelist=timelist
        self.logger.info(f'timecount:{timecount},timelist:{timelist}')
        
        timeposdict={time:i for i,time in enumerate(timelist)}
        time_arraylist=[]
        for time in timelist:
            d_idx=whichdictdict[time]
            datadict=datadictlist[d_idx]
            thistime_idxlist=[idx for idx,idx_time in enumerate(datadict[timevar]) if idx_time==time]
            obsdata=[[datadict[varkey][idx] for varkey in varlist]for idx in thistime_idxlist]
            time_arraylist.append(np.array(obsdata))
        self.logger.info(f'time_arraylist shapes:{[_array.shape for _array in time_arraylist]}')
        return time_arraylist
            
            
    def unpackOrderedDict(self,odict):
        pydict={}
        rowcount=len(odict)
        for r_idx,row in enumerate(odict):
            for key in row:
                val=row[key]
                if key in pydict:
                    pydict[key][r_idx]=val
                else: 
                    pydict[key]=[None for _ in range(rowcount)]
                    pydict[key][r_idx]=val
        return pydict
        
    def makeTSHistogram(self,varlist=None,yearlist=None):
        try: self.datadictlist
        except:self.pickleDataDictList() 
        if not varlist:
            varlist=[
                'sale_year',
                'saleprice',
                'assessedvalue',
                'secchi',
                'wateraccess',
                'bayfront',
                'waterhouse',
                'shorelinedistance' 
                ]
        time_arraylist=self.doDictListToNpTS(self.datadictlist,varlist,timevar='sale_year') #(time,obs,var) #also creates self.timelist
        t_idx=varlist.index('sale_year')
        timelenlist=[_array.shape[0] for _array in time_arraylist]
        xlabel=f'Sale Years {self.timelist[0]}-{self.timelist[-1]}'
        ylabel='frequency'
        title='Frequency of Sales by Year'
        self.my2dbarplot(
            np.array(self.timelist),
            np.array(timelenlist),
            xlabel=xlabel,ylabel=ylabel,title=title)
        #self.fig
      
