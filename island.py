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
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.figdict={}
        self.TSHistogramlist=[]
        self.datadir=os.path.join(os.getcwd(),'data')
        self.helper=Helper()
        self.datadir=os.path.join(os.getcwd(),'data')
        self.datadictlistpath=os.path.join(self.datadir,'datadictlist.pickle')
        '''self.var_type_dict={
            'sale_year':np.int16,
            'saleprice':np.int64,
            'assessedvalue':np.int64,
            'secchi':np.float64,
            'wateraccess':np.bool,
            'bayfront':np.bool,
            'waterhouse':np.bool,
            'shorelinedistance':np.float32
        }
        self.varlist=[key for key in self.var_type_dict]'''
        self.varlist=[
            'sale_year',
            'saleprice',
            'assessedvalue',
            'secchi',
            'wateraccess',
            'bayfront',
            'waterhouse',
            'shorelinedistance'
            ]
        self.fig=None;self.ax=None
        DataView.__init__(self)
        
        
    def make2dHistogram(self,):   
        try:self.time_arraylist
        except: self.makeTimeArrayList()
        time_arraylist=self.time_arraylist
        varlist=self.varlist
        t_idx=varlist.index('sale_year')
        timelenlist=[_array.shape[0] for _array in self.time_arraylist]
        xlabel=f'Sale Years {self.timelist[0]}-{self.timelist[-1]}'
        ylabel='Count'
        title='Count of Sales by Year'
        fig=self.my2dbarplot(
            np.array(self.timelist,dtype=np.int16),
            np.array(timelenlist),
            xlabel=xlabel,ylabel=ylabel,title=title,fig=None,subplot_idx=(1,1,1))
        if not 'hist2d' in self.figdict:
            self.figdict['hist2d']=[fig]
        else:
            self.figdict['hist2d'].append(fig)
    
    def makeIndividualTSHistogram(self,varlist=None):
        if not varlist:
            varlist=self.varlist
        for var in varlist:
            self.makeTSHistogram(varlist=[var])
        
    
    def makeTSHistogram(self,varlist=None,combined=1):
        try:self.time_arraylist
        except: self.makeTimeArrayList()
        time_arraylist=self.time_arraylist
        if not varlist:
            varlist=self.varlist
            var_idx_list=list(range(len(varlist)))
        else:
            var_idx_list=[self.varlist.index(var) for var in varlist]
        if 'sale_year' in varlist:
            varcount=len(varlist)-1
        else:varcount=len(varlist)
        fig=plt.figure(figsize=[14,varcount*12])
        
        subplot_idx=[varcount,1,1]
        for idxidx,var in enumerate(varlist):
            if not var=='sale_year':
                var_idx=var_idx_list[idxidx]
                if not combined:
                    subplot_idx=[1,1,1]
                    fig=None
                fig,histdict=self.my3dHistogram([nparray[:,var_idx] for nparray in time_arraylist],var,subplot_idx=subplot_idx,fig=fig)
                self.TSHistogramlist.append({'histTS':histdict})
                    
                subplot_idx[2]+=1
        if not 'histTS' in self.figdict:
            self.figdict['histTS']=[fig]
        else:
            self.figdict['histTS'].append(fig)
      
        
    def doDictListToNpTS(self,datadictlist,timevar='sale_year'):
        '''
        timelist dims(obs,var)
        '''
        varlist=self.varlist
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
            time_arraylist.append(np.array(obsdata,dtype=np.float64))
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
        
    def makeTimeArrayList(self,):
        try: self.datadictlist
        except:self.pickleDataDictList() 
        time_arraylist=self.doDictListToNpTS(self.datadictlist,timevar='sale_year') #(time,obs,var) #also creates self.timelist
        self.time_arraylist=time_arraylist
    
    
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
        '''
        saves if  datadictlist
        else loads
        '''
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