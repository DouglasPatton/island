import numpy as np
import pandas as pd
import csv
import pickle,json
import os
import logging
from helpers import Helper
import matplotlib.pyplot as plt
#import cpi #https://github.com/datadesk/cpi
from data_viz import DataView
#from datetime import date


class IslandData(DataView):
    def __init__(self,dpi=100):
        self.dpi=dpi
        self.helper=Helper()
        logdir=os.path.join(os.getcwd(),'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=os.path.join(logdir,f'island.log')
        logpath=os.path.join(logdir,handlername)
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(logpath, maxBytes=10**6, backupCount=20)],
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.figdict={}
        self.TSHistogramlist=[]
        self.datadir=os.path.join(os.getcwd(),'data')
        self.printdir=os.path.join(os.getcwd(),'print')
        if not os.path.exists(self.printdir):
            os.mkdir(self.printdir)
        self.datadictlistpath=os.path.join(self.datadir,'datadictlist.pickle')
        self.vardict={
            'sale_year':np.uint16,'saleprice':np.int64,'assessedvalue':np.int64,
            'postsandy':np.uint16,'secchi':np.float64,
            'wqbayfront':np.float64,'wqwateraccess':np.float64,'wqwaterhouse':np.float64,
            'totalbathroomsedited':np.float64,'totallivingarea':np.float64,'saleacres':np.float64,
            'distance_park':np.float64,'distance_nyc':np.float64,'distance_golf':np.float64,
            'wqshorelinedistancedv3_1000':np.float64,'wqshorelinedistancedv1000_2000':np.float64,
            'wqshorelinedistancedv2000_3000':np.float64,'wqshorelinedistancedv3000_4000':np.float64,
            'education':np.float64,'income':np.float64,'povertylevel':np.float64,
            'pct_white':np.float64,'pct_asian':np.float64,'pct_black':np.float64,
            'bayfront':np.uint16,'wateraccess':np.uint16,'waterhouse':np.uint16,
            'shorelinedistance':np.uint16,'distance_shoreline':np.float64, 
            'shorelinedistancedv3_1000':np.uint16,'shorelinedistancedv1000_2000':np.uint16,
            'shorelinedistancedv2000_3000':np.uint16,'shorelinedistancedv3000_4000':np.uint16,
            'soldmorethanonceinyear':np.uint16,'soldmorethanonceovertheyears':np.uint16,
            'latitude':np.float64,'longitude':np.float64            
            }
        self.varlist=[var for var in self.vardict]
        self.geogvars=['latitude','longitude']
        self.dollarvars=['saleprice','assessedvalue','income']
        self.fig=None;self.ax=None
        self.figheight=10;self.figwidth=10
        DataView.__init__(self)
    
    
    def getCPI(self,to_year=2015):
        cpi_factors_path=os.path.join(self.datadir,f'cpi_factors-{to_year}.json')
        try:
            with open(cpi_factors_path,'r') as f:
                cpi_factors=json.load(f)
            return cpi_factors
        except:
            self.logger.info('building cpi_factors')
        import cpi
        cpi_factor_list=[]
        for t in self.timelist:#
            cpi_factor=np.float64(cpi.inflate(1,int(t),to=to_year))
            cpi_factor_list.append(cpi_factor)
        with open(cpi_factors_path,'w') as f:
            json.dump(cpi_factor_list,f)
        return cpi_factor_list
    
    def addRealByCPI(self,to_year=2015):
        try:self.time_arraytup
        except: self.makeTimeListArrayList()
            
        try:
            dollarvarcount=self.dollarvars
            deflated_array_list=[]
            timelist_arraylist,varlist=self.time_arraytup
            cpi_factor_list=self.getCPI(to_year=2015)
            for t in range(len(timelist_arraylist)):
                nparraylist=timelist_arraylist[t]
                
                #deflated_var_array=np.empty(nparray.shape[0],dollarvarcount,dtype=np.float64)
                for dollarvar in self.dollarvars:
                    var_idx=varlist.index(dollarvar)
                    cpi_factor=cpi_factor_list[t]
                    real_dollar_array=nparraylist[var_idx]*cpi_factor
                    nparraylist.append(real_dollar_array)
                    #nparray=np.concatenate([nparray,np.float64(nparray[:,var_idx][:,None])*cpi_factor],axis=1) 
                #timelist_arraylist[t]=nparray
            for var in self.dollarvars:#separate loop since just happens once per t
                varlist.append(var+'_real-'+str(to_year))
            self.logger.info(f'np.shape for timelist_arraylist:{[nparray.shape for arraylist in timelist_arraylist for nparray in arraylist]}')
            self.varlist=varlist
            self.time_arraytup=(timelist_arraylist,varlist)
            return
        except:
            self.logger.exception('')
    
            
    def arrayListToPandasDF(self,):
        try:self.time_arraytup
        except: self.makeTimeListArrayList()
        timelist_arraylist,varlist=self.time_arraytup
        [nparraylist.append(np.arange(nparraylist[0].shape[0])) for nparraylist in timelist_arraylist]
        varlist.append('idx') # name the new column 'idx'                                           
        columnlist=[np.concatenate([nparraylist[varidx] for nparraylist in timelist_arraylist],axis=0) for varidx in range(len(varlist))]
        #timelist_arraylist_indexed=[np.concatenate([nparray,np.arange(nparray.shape[0])[:,None]],axis=1) for nparray in timelist_arraylist] # add a new 'column' of data just numbering from 0 for pandas multi-index
        
        #full2darray=np.concatenate(timelist_arraylist_indexed,axis=0)
        indexvarlist=['postsandy','sale_year','idx']
        index=[columnlist[varlist.index(idxvar)] for idxvar in indexvarlist]
        self.logger.info(f'index:{index}')
        index=[nparray[:,None] for nparray in index]
        index=np.concatenate(index,axis=1)
        index_df=pd.DataFrame(data=index,columns=indexvarlist)
        multi_index=pd.MultiIndex.from_frame(index_df)
        pd_data_dict={varlist[i]:columnlist[i] for i in range(len(varlist))}
        self.df=pd.DataFrame(pd_data_dict,index=multi_index)
       
    
        
    def printDFtoSumStats(self,df=None,varlist=None):
        pd.set_option('display.max_colwidth', None)
        printpath=os.path.join(self.printdir,'sumstats.html')
        if df is None:
            df=self.df
        if varlist is None:
            varlist=self.varlist
        
        levels=['Pre-Sandy','Post-Sandy']
        html_list=[]
        for level in df.index.levels[0]:
            html_list.append(levels[level]+'<br>'+df.loc[level].describe().to_html())
        
        self.sumstats_html='<br>'.join(html_list)
        with open(printpath,'w') as f:
            f.write(self.sumstats_html) 
        
        
    def make2dHistogram(self,):   
        try:self.time_arraytup
        except: self.makeTimeListArrayList()
        timelist_arraylist,varlist=self.time_arraytup
        #varlist=self.varlist
        #t_idx=varlist.index('sale_year')
        timelenlist=[_array.shape[0] for _array in timelist_arraylist]
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
        except: 
            self.logger.exception('self.time_arraytup error, activating makeTimeListArrayList')
            self.makeTimeListArrayList()
        
        timelist_arraylist,time_array_varlist=self.time_arraytup
        self.logger.info(f'makeTSHistogram varlist:{varlist}')
        self.logger.info(f'len(time_array_varlist):{len(time_array_varlist)}, time_array_varlist:{time_array_varlist}')
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
                    
                    fig,histdict=self.my3dHistogram([nparraylist[var_idx] for nparraylist in timelist_arraylist],
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
            if not 'postsandy' in varlist:
                   self.varlist.append('postsandy')
        if not 'postsandy' in varlist:
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
        timelist_arraylist=[]
        for time in timelist:
            thistime_idxlist=[]
            for d_idx in whichdictdict[time]:
                datadict=datadictlist[d_idx]
                thisN=len(datadict[timevar])
                datadict['postsandy']=[d_idx for _ in range(thisN)]

                thistime_idxlist.extend([(d_idx,idx) for idx,idx_time in enumerate(datadict[timevar]) if idx_time==time])
            vararraylist=[]
            for var in varlist:
                vartype=self.vardict[var]
                column=[datadictlist[d_idx][var][idx] for d_idx,idx in thistime_idxlist] 
                colarray=np.array(column,dtype=vartype) #columns vary along dim0
                self.logger.info(f'for time:{time}, colarray.shape,type:{(colarray.shape,colarray.dtype)}')
                vararraylist.append(colarray)
            '''for d_idx,idx in thistime_idxlist:
                idxdatalist=[]
                for varkey in varlist:
                    idxdatalist.append(datadictlist[d_idx][varkey][idx])
                    #self.logger.info(f'datalist:{datalist}')
                    newarray=np.array(datalist,dtype=self.vardict[varkey])
                    arraylist.append(newarray)
                    self.logger.info(f'newarray.shape:{newarray.shape}, newarray.dtype:{newarray.dtype}')
                arraylist=[newarray[:,None] for newarray in arraylist]
                array_t=np.concatenate(arraylist,axis=1)
                arraylistlist.append(array_t)'''
            #obsdata=np.concatenate(vararraylist,axis=1, dtype=object)
            obsdata=vararraylist
            #self.logger.info(f'for time:{time}, obsdata.shape:{obsdata.shape}')
            timelist_arraylist.append(obsdata)
            #obsdata=np.concatenate([np.concatenate([np.array(datadictlist[d_idx][varkey][idx],dtype=self.vardict[varkey])[:,None] for varkey in varlist],axis=1) for d_idx,idx in thistime_idxlist],axis=0)
            #timelist_arraylist.append(obsdata)
                    
            #obsdata=[[datadictlist[d_idx][varkey][idx] for varkey in varlist]for d_idx,idx in thistime_idxlist]
            #timelist_arraylist.append(np.array(obsdata,dtype=np.float64))
        self.logger.info(f'timelist_arraylist shapes:{[_array.shape for arraylist in timelist_arraylist for _array in arraylist]}')
        return timelist_arraylist,varlist
    
    
            
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
        
    def makeTimeListArrayList(self,varlist=None):
        if varlist is None:
            varlist=self.varlist
        try: self.datadictlist
        except:self.pickleDataDictList() 
        timelist_arraylist,varlist=self.doDictListToNpTS(self.datadictlist,timevar='sale_year',varlist=varlist) #(time,obs,var) #also creates self.timelist
        self.time_arraytup=(timelist_arraylist,varlist)
    
    
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