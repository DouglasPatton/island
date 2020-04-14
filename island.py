import numpy as np
import csv
import pickle
import os
import logging
from helpers import Helper
import matplotlib.pyplot as plt
import cpi #https://github.com/datadesk/cpi
from data_viz import DataView
#from datetime import date


class IslandData(DataView):
    def __init__(self,dpi=100):
        self.dpi=dpi
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
            'postsandy',
            'secchi',
            'wqbayfront',
            'wqwateraccess',
            'wqwaterhouse',
            'totalbathroomsedited',
            'totallivingarea',
            'saleacres',
            'distance_park',
            'distance_nyc',
            'distance_golf',
            'wqshorelinedistancedv3_1000',
            'wqshorelinedistancedv1000_2000',
            'wqshorelinedistancedv2000_3000',
            'wqshorelinedistancedv3000_4000',
            'education',
            'income',
            'povertylevel',
            'pct_white',
            'pct_asian',
            'pct_black',
            'bayfront',
            'wateraccess',
            'waterhouse',
            'shorelinedistance',
            'shorelinedistancedv3_1000',
            'shorelinedistancedv1000_2000',
            'shorelinedistancedv2000_3000',
            'shorelinedistancedv3000_4000',
            'distance_shoreline',
            'soldmorethanonceinyear',
            'soldmorethanonceovertheyears',
            'latitude',
            'longitude'            
            ]
        self.geogvars=['latitude','longitude']

        
        
        
        self.dollarvarlist=['saleprice','assessedvalue','income']
        self.fig=None;self.ax=None
        self.figheight=10;self.figwidth=10
        DataView.__init__(self)
    
    def addRealByCPI(self,to_year=2015):
        try:self.time_arraytup
        except: self.makeTimeArrayList()
        dollarvarcount=self.dollarvarlist
        deflated_array_list=[]
        time_arraylist,varlist=self.time_arraytup
        for t in range(len(time_arraylist)):
            nparray=time_arraylist[t]
            from_year=self.timelist[t]
            cpi_factor=cpi.inflate(1,int(from_year),to=to_year)
            #deflated_var_array=np.empty(nparray.shape[0],dollarvarcount,dtype=np.float64)
            for dollarvar in self.dollarvarlist:
                var_idx=varlist.index(dollarvar)
                np.concatenate([nparray,nparray[:,var_idx][:,None]*cpi_factor],axis=1) 
        for var in self.dollarvarlist:
            varlist.append(var+'_real-'+str(to_year))
        self.varlist=varlist
        self.time_arraytup=(time_arraylist,varlist)
            
        
        
    def make2dHistogram(self,):   
        try:self.time_arraytup
        except: self.makeTimeArrayList()
        time_arraylist,varlist=self.time_arraytup
        #varlist=self.varlist
        #t_idx=varlist.index('sale_year')
        timelenlist=[_array.shape[0] for _array in time_arraylist]
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
        try:self.time_arraytup
        except: self.makeTimeArrayList()
        time_arraylist,time_array_varlist=self.time_arraytup
        self.logger.info(f'makeTSHistogram varlist:{varlist}')
        if not varlist:
            varlist=time_array_varlist
            var_idx_list=list(range(len(varlist)))
        else:
            var_idx_list=[time_array_varlist.index(var) for var in varlist]
        if 'sale_year' in varlist:
            varcount=len(varlist)-1
        else:varcount=len(varlist)
        if combined:
            fig=plt.figure(dpi=self.dpi,figsize=[self.figwidth*2,self.figheight*varcount])
            
        figlist=[]
        subplot_idx=[varcount,2,1]
        for idxidx,var in enumerate(varlist):
            if not var=='sale_year':
                var_idx=var_idx_list[idxidx]
                if not combined:
                        subplot_idx=[1,2,1]
                        fig=None
                for norm_hist in [0,1]:
                    
                    fig,histdict=self.my3dHistogram([nparray[:,var_idx] for nparray in time_arraylist],
                                                    var,self.timelist,subplot_idx=subplot_idx,fig=fig,norm_hist=norm_hist)
                    self.TSHistogramlist.append({'histTS':histdict})
                    subplot_idx[2]+=1
                figlist.append(fig)
        if not 'histTS' in self.figdict:
            self.figdict['histTS']=figlist
        else:
            self.figdict['histTS'].extend(figlist)
      
        
    def doDictListToNpTS(self,datadictlist,timevar='sale_year',varlist=None):
        '''
        timelist dims(obs,var)
        '''
        if varlist is None:
            varlist=self.varlist
        self.varlist.append('postsandy')
        varlist.append('postsandy')
        timelist=[];whichdictdict={}
        for d_idx,datadict in enumerate(datadictlist):
            for time in datadict[timevar]:
                if not time in whichdictdict:
                    whichdictdict[time]=[d_idx]
                else: 
                    if d_idx not in whichdictdict[time]:
                        whichdictdict[time].append(d_idx)
                if not time in timelist:
                    timelist.append(time)
        timecount=len(timelist)
        timelist.sort()
        self.timelist=timelist
        self.logger.info(f'timecount:{timecount},timelist:{timelist}')
        timeposdict={time:i for i,time in enumerate(timelist)}
        time_arraylist=[]
        for time in timelist:
            thistime_idxlist=[]
            for d_idx in whichdictdict[time]:
                datadict=datadictlist[d_idx]
                thisN=len(datadict[timevar])
                datadict['postsandy']=[d_idx for _ in range(thisN)]

                thistime_idxlist.extend([(d_idx,idx) for idx,idx_time in enumerate(datadict[timevar]) if idx_time==time])
            
            
                    
            obsdata=[[datadictlist[d_idx][varkey][idx] for varkey in varlist]for d_idx,idx in thistime_idxlist]
            time_arraylist.append(np.array(obsdata,dtype=np.float64))
        self.logger.info(f'time_arraylist shapes:{[_array.shape for _array in time_arraylist]}')
        return time_arraylist,varlist
    
    
            
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
        
    def makeTimeArrayList(self,varlist=None):
        if varlist is None:
            varlist=self.varlist
        try: self.datadictlist
        except:self.pickleDataDictList() 
        time_arraylist,varlist=self.doDictListToNpTS(self.datadictlist,timevar='sale_year',varlist=varlist) #(time,obs,var) #also creates self.timelist
        self.time_arraytup=(time_arraylist,varlist)
    
    
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