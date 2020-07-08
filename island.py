
import numpy as np
import pandas as pd
import csv
import pickle,json
import os
import logging
from logging import handlers
from helpers import Helper
import matplotlib.pyplot as plt
#import cpi # imported conditionally, later on #https://github.com/datadesk/cpi
from data_viz import DataView
from datetime import datetime
from models import SpatialModel
from math import floor,log10,ceil
import re

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
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)
        self.klist=[2]#[2,4,6]#[25,50,100]
        self.resultsdictlist=[]
        self.figdict={}
        self.sumstatsdict={}
        self.TSHistogramlist=[]
        self.resultsDFdictlist=[]
        cwd=os.getcwd()
        self.datadir=os.path.join(cwd,'data')
        self.resultsdir=os.path.join(cwd,'results');
        if not os.path.exists(self.resultsdir):os.mkdir(self.resultsdir)
        self.resultspath=os.path.join(self.resultsdir,'resultsdictlist.pickle')
        
        self.printdir=os.path.join(cwd,'print')
        if not os.path.exists(self.printdir):
            os.mkdir(self.printdir)
        
        self.datadictlistpath=os.path.join(self.datadir,'datadictlist.pickle')
        self.yeardummydict={f'dv_{i}':np.uint16 for i in range(2002,2016)}
        self.yeardummylist=[key for key in self.yeardummydict]
        self.vardict,self.modeldict,self.std_transform=self.setmodel()
        self.varlist=[var for var in self.vardict]
        self.geogvars=['latitude','longitude']
        
        
        self.fig=None;self.ax=None
        self.figheight=10;self.figwidth=10
        DataView.__init__(self)
        
        
    def setmodel(self):
        std_transform=[]#['saleprice','assessedvalue','totallivingarea','parcel_area','distance_park','distance_nyc','distance_golf','income','distance_shoreline']
        vardict={
            'sale_year':np.uint16,'saleprice':np.int64,'assessedvalue':np.int64,
            'postsandy':np.uint16,'secchi':np.float64,
            'wqbayfront':np.float64,'wqwateraccess':np.float64,'wqwaterhouse':np.float64,
            'totalbathroomsedited':np.float64,'totallivingarea':np.float64,'parcel_area':np.float64,
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
        vardict={**vardict, **self.yeardummydict}

        xvarlist=[
            'secchi','bayfront','wateraccess','wqbayfront','wqwateraccess',
            'totalbathroomsedited','totallivingarea','parcel_area',
            'distance_park','distance_nyc','distance_golf',
            'shorelinedistancedv3_1000','shorelinedistancedv1000_2000',
            'shorelinedistancedv2000_3000',
            'wqshorelinedistancedv3_1000','wqshorelinedistancedv1000_2000',
            'wqshorelinedistancedv2000_3000',
            'education','income_real-2015','povertylevel','pct_white']
        xvarlist.extend(self.yeardummylist)
        
        modeldict={
                'combine_pre_post':0,
                'period':None, # used later to record which time period is included in data for a model
                'modeltype':'OLS',#,'GM_Error_Het',#'GM_Lag',#'SLM',#'SEM',#'SLM',
                'klist':self.klist,
                #'crs':'epsg:4326',
                'xvars':xvarlist,
                'yvar':'saleprice_real-2015',
                'transform':{'ln_wq':0,'ln_y':1},
                'wt_type':'NN',#'inverse_distance_NN_exp2',#'inverse_distance_NN',
                #'inverse_distance_nn_exp2',#'inverse_distance_NN_exp1',#'inverse_distance',#
                'wt_norm':'rowsum',#'rowmax',#'doublesum',#
                'NNscope':'year',#'period',#     # 'period' groups obs by pre or post, 'year' groups by year
                'distmat':1, # 0 if not enough ram to do all NN at once
                'distance':[50,75,100,150,200,300,400,600,800,1200,1600,3200], #'default'
                'cpi_dollar_vars':{
                    'saleprice':{'area':"New York-Newark-Jersey City, NY-NJ-PA",'items':"Housing"},
                    'income':{'area':"New York-Newark-Jersey City, NY-NJ-PA"}
                }

            }
        return vardict,modeldict,std_transform
    
    
    def doDistanceVars(self):
        modeldict=self.modeldict
        df=self.df
        xvarlist=modeldict['xvars']
        
        try:
            distance_param=modeldict['distance']
        except KeyError:
            distance_param='default'
        except:
            assert False, 'unexpected'
        if distance_param=='default':
            return 
        if type(distance_param) is list:
            newxvarlist=[]
            for xvar in xvarlist:
                if not re.search('shorelinedistance',xvar):
                    newxvarlist.append(xvar) 
            self.df,self.modeldict['xvars']=self.buildDistanceVarsFromList(df,distance_param,newxvarlist)
            
        else: assert False, 'not developed'
        
    def buildDistanceVarsFromList(self,df,cutlist,xvarlist):
        raw_dist=df.loc[(slice(None),),'distance_shoreline'].astype(float)
        raw_wq=df.loc[(slice(None),),'secchi'].astype(float)
        max_d=raw_dist.max()
        cutlist.sort()
        if cutlist[-1]<max_d:
            cutlist=cutlist+[ceil(max_d)]
        if cutlist[0]!=0:
            cutlist=[0]+cutlist
        wateraccess=df.loc[(slice(None),),'wateraccess']
        bayfront=df.loc[(slice(None),),'bayfront']
        for idx in range(len(cutlist)-2): # -1 b/c left and right, -2 to omit 1 dv
            newvar=f'Distance to Shoreline {cutlist[idx]}m-{cutlist[idx+1]}m'
            left=cutlist[idx];right=cutlist[idx+1]
            df.loc[(slice(None),),newvar]=0
            df.loc[raw_dist>=left,newvar]=1
            df.loc[raw_dist>right,newvar]=0
            df.loc[wateraccess==1,newvar]=0
            df.loc[bayfront==1,newvar]=0
            
            xvarlist.append(newvar)
            newvar_wq='secchi*'+newvar
            df.loc[(slice(None),),newvar_wq]=df.loc[(slice(None),),newvar]*raw_wq
            xvarlist.append(newvar_wq)
            #print(df,xvarlist)
        
        return df,xvarlist
    
    
    def runSpatialModel(self,modeldict=None,justW=0):
        if modeldict is None:
            modeldict=self.modeldict
        spatial_model_tool_tool=SpatialModel(modeldict)
        if justW:
            spatial_model_tool.justMakeWeights(df=self.df)
            return
        resultdictlist=spatial_model_tool_tool.run(df=self.df)
        self.resultsdictlist.extend(resultdictlist)
        self.saveSpatialModelResults(resultdictlist)
        
        #return self.resultsdictlist
    
    def saveSpatialModelResults(self,resultsdict,load=0):
        if load:
            with open('resultsdictlist.pickle','rb') as f:
                resultsdictlist=pickle.load(f)
            self.resultsdictlist.extend(resultsdictlist)
            return resultsdictlist
        else:
            try:
                with open('resultsdictlist.pickle','rb') as f:
                    oldresults=pickle.load(f)
                oldresults.extend(resultsdict)
                resultsdictlist=oldresults
            except:
                resultsdictlist=[resultsdict]
            with open('resultsdictlist.pickle','wb') as f:
                pickle.dump(resultsdictlist,f)
        return
  

    def flattenListList(self,listlist):
        outlist=[]
        for l in listlist:
            if type(l) is list:
                flatlist=self.flattenListList(l)
            else:
                flatlist=[l]
            outlist.extend(flatlist)
        return outlist
     
    
    
        

    def doModelResultsToDF(self,):
        try: 
            assert self.resultsdictlist,"results not loaded, loading results"
            resultsdictlist=self.resultsdictlist
        except: 
            resultsdictlist=self.saveSpatialModelResults([],load=1)
        
        #flatresults=self.flattenListList(results)
        
        for resultsdict in resultsdictlist:
            modeldict=resultsdict['modeldict']
            modelresult=resultsdict['results']
            
        
            resultsdata={}
            resultsdata['xvarlist']=np.array(modelresult.name_x)
            resultsdata['betalist']=np.array(modelresult.betas).flatten()
            resultsdata['stderrlist']=np.array(modelresult.std_err)
            zstatlist,pvallist=zip(*modelresult.z_stat)
            resultsdata['zstatlist']=np.array(zstatlist)
            resultsdata['pvallist']=npmodeldict.array(pvallist)
            self.logger.info(f'resultsdata:{resultsdata}')
            resultsdf=pd.DataFrame(resultsdata)
            resultDFdict={'modeldict':modeldict,'resultsdf':resultsdf}
            self.resultsDFdictlist.append(resultDFdict)
    
    def myFlatDict(self, complexdict, keys=None):
        thistype = type(complexdict)
        if not thistype is dict:
            return {'val': complexdict}
        if keys == None and thistype is dict:
            keys = [key for key, val in complexdict.items()]
        flatdict = {}
        for key in keys:
            try:
                val = complexdict[key]
            except:
                val = 'no val found'
            newdict = self.myFlatDict(val)
            for key2, val2 in newdict.items():
                flatdict[f'{key}:{key2}'] = [val2]
        return flatdict
    
    def simplifyDict(self,adict):
        sdict={};splitchar=','
        for key,val in adict.items():
            if type(val) is dict:
                sval=splitchar.join([f'{key}-{val}' for key,val in val.items()])
            elif type(val) in [tuple,list]:
                sval=splitchar.join(val)
            else:
                sval=val
            sdict[key]=sval
        return sdict
    
    def createWQGraph(self):
        
        try: 
            assert self.resultsdictlist,"results not loaded, loading results"
            resultsdictlist=self.resultsdictlist
        except: 
            resultsdictlist=self.saveSpatialModelResults([],load=1)
        I=len(resultsdictlist)
        summary_text='Model Summaries\n'+f'for {I}  models\nPrinted on {datetime.now()}\n'
        
        
        resultsdictflatlist=self.flattenListList(resultsdictlist)
        p0=0;p1=0
        i=-1
        while not (p0 and p1):
            i+=1
            resultsdict=resultsdictflatlist[i]
            modeldict=resultsdict['modeldict']
            p=modeldict['period']
            m=modeldict['modeltype']
            if m.lower()=='ols':
                if p==0 and p0==0:
                    p0=resultsdict
                if p==1 and p1==0:
                    p1=resultsdict
        fig=plt.figure(dpi=600,figsize=[8,6])
        plt.xticks(rotation=17)
        ax=fig.add_subplot()
        ax.set_title('Fixed Effects Estimates of % Increase in Sale Price from 1m Increase in Water Clarity')#Fixed Effects Estimates for Water Clarity by Distance from Shore Band')
        ax.set_xlabel('Distance from Shore Bands (not to scale)')
        ax.set_ylabel('Partial Derivatives of Sale Price by Water Clarity by Distance from Shore Band')
        
        self.extractAndPlotWQ(p0,ax,'Pre-Sandy',color='r',hatch='.'*5,ls='--')
        self.extractAndPlotWQ(p1,ax,'Post-Sandy',color='g',hatch=None,ls='-')
        ax.legend(loc=1)
        ax.margins(0)
        figpath=self.helper.getname(os.path.join(self.printdir,'wq_graph.png'))
        fig.savefig(figpath)
        
    
    def extractAndPlotWQ(self,resultsdict,ax,plottitle,color='b',hatch='x',ls='-'):
        results=resultsdict['results']
        #modeldict=resultsdict['modeldict']
        #xvarlist=modeldict['xvars']
        wqvars_idx,wqvars=zip(*[[idx,var] for idx,var in enumerate(results.name_x) if re.search('secchi\*distance',var.lower()) or re.search('secchi\*bayfront',var.lower())])
        wqvars_idx=list(wqvars_idx)
        wqvars=list(wqvars)
        for idx,var in enumerate(results.name_x):
            if var.lower()=='secchi':
                wqvars_idx.append(idx)
                wqvars.append('3200m-4000m')
                secchi=results.betas[wqvars_idx][0]*100
        #print(wqvars_idx,wqvars)
        #print('betas',results.betas)
        #print('std_err',results.std_err)
        
        wqcoefs=[results.betas[idx][0]*100 for idx in wqvars_idx] # b/c pysal returns each beta inside its own list. *100 b/c pct. 
        wqcoef_stderrs=[results.std_err[idx]*100 for idx in wqvars_idx] # 100 b/c pct.
        wqcoef_names=[]
        for var in wqvars: # a loop for shortening names
            if re.search('secchi\*distance',var.lower()):
                wqcoef_names.append(var[29:])
            elif re.search('secchi\*bayfront',var.lower()):
                wqcoef_names.append(var[7:])
            else:
                wqcoef_names.append(var)
        bcount=len(wqcoefs)
        for b in range(bcount-1):# add global constant to all terms except last one, which was the omitted variable
            wqcoefs[b]=wqcoefs[b]+wqcoefs[-1]
            wqcoef_stderrs[b]=wqcoef_stderrs[b]+wqcoef_stderrs[-1]
            
            
        
                
            
        
        self.logger.info(f'{[wqcoef_names,wqcoefs,wqcoef_stderrs ]}')
        self.makePlotWithCI(wqcoef_names,wqcoefs,wqcoef_stderrs,ax,plottitle=plottitle,color=color,hatch=hatch,ls=ls)
        
                    
            
        
        
    def printModelSummary(self,stars=None):
        try: 
            assert self.resultsdictlist,"results not loaded, loading results"
            resultsdictlist=self.resultsdictlist
        except: 
            resultsdictlist=self.saveSpatialModelResults([],load=1)
        I=len(resultsdictlist)
        summary_text='Model Summaries\n'+f'for {I}  models\nPrinted on {datetime.now()}\n'
        
        
        resultsdictflatlist=self.flattenListList(resultsdictlist)
        
        for resultsdict in resultsdictflatlist:
            modeldict=resultsdict['modeldict']
            modelresult=resultsdict['results']
            summary_text+=str(modeldict)+'\n'
            if stars:
                try:
                    pvals=[i[1] for i in modelresult.z_stat]
                except AttributeError:
                    pvals=[i[1] for i in modelresult.t_stat]
                except:
                    assert False, 'unexpected'
                betas=[i[0] for i in modelresult.betas] # b/c each beta is stored as a list with 1 item
                summary_text+=self.doStars(modelresult.name_x,betas,pvals)+'\n\n\n'
            else:    
                summary_text+=modelresult.summary+'\n\n\n'

        
        if stars:
            savepath='modelresults_stars.txt'
        else:
            savepath='modelresults.txt'
        savepath=self.helper.getname(savepath)
        with open(savepath,'w') as f:
            f.write(summary_text)
            
    def doStars(self,names,betas,pvals,round_digits=5):        
        text=''
        if round_digits:
            betas=[self.round_sig(beta,round_digits) for beta in betas]
        for i in range(len(names)):
            text+=f'{names[i]},{betas[i]}{self.starFromPval(pvals[i])}\n'
        return text
    
    def round_sig(self,x, dig=2):
        if x==0: return x
        return round(x, dig-int(floor(log10(abs(x))))-1)
    
    def starFromPval(self,pval):
        rounded_pval=round(pval,2)
        if rounded_pval>0.1:
            return ''
        elif rounded_pval>0.5:
            return '*'
        elif rounded_pval>0.01:
            return '**'
        else:
            return '***'
            
            
    
    def printModelResults(self,):
        try:
            assert self.resultsDFdictlist,"building resultsDFdictlist"
        except:
            self.doModelResultsToDF()
        resultsDFdictlist=self.resultsDFdictlist
        modeltablehtml='Model Results'
        I=len(resultsDFdictlist)
        for i,resultsDFdict in enumerate(resultsDFdictlist):
            
            modeldict=resultsDFdict['modeldict']
            simple_modeldict=self.myFlatDict(modeldict)
            print(f'simple_modeldict:{simple_modeldict}')
            #title=f"----{modeldict['period']}Sandy,k={modeldict['klist']}----""
            titledf=pd.DataFrame(simple_modeldict).T

            title=titledf.to_html()
            df=resultsDFdict['resultsdf']
            result_html=df.to_html()
            modeltablehtml+='<br><br>'+f'model #{i+1} of {I}<br>'+title+'<br>'+result_html
            
        with open('semresults.html','w') as f:
            f.write(modeltablehtml)
            
    
    def changeVarlistWQtoSecchi(self,varlist):
        newlist=[]
        for var in varlist:
            if var[0:2]=='wq':
                newlist.append('secchi*'+var[2:])
            else:newlist.append(var)
        
        return newlist
    
            
    def arrayListToPandasDF(self,):
        try:self.time_arraytup
        except: self.makeTimeListArrayList()
        timelist_arraylist,varlist=self.time_arraytup
        varlist=self.changeVarlistWQtoSecchi(varlist)
        self.modeldict['xvars']=self.changeVarlistWQtoSecchi(self.modeldict['xvars'])
        [nparraylist.append(np.arange(nparraylist[0].shape[0])) for nparraylist in timelist_arraylist] #appending a column to serve as idx
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
        self.df_raw=pd.DataFrame(pd_data_dict,index=multi_index)
        self.df=self.doStandardizeDF(self.df_raw)
    
    def doStandardizeDF(self,df):
        df_t=df.apply(self.doSeriesSumStats,axis=0)
        return df_t
        
    def doSeriesSumStats(self,aseries):
        varname=aseries.name
        if varname not in self.std_transform:
            self.logger.info(f'varname:{varname} not in std_transform...pass')
            return aseries
        if varname not in self.sumstatsdict:
            smean=aseries.mean()
            sstd=aseries.std()
            self.sumstatsdict[varname]={'mean':smean,
                                        'std':sstd}
            aseries=(aseries-smean)/sstd
            self.logger.info(f'varname:{varname} standardized. mean:{smean},std:{sstd}')
            return aseries
        else:
            self.logger.info(f'varname:{varname} already standardized')
            return aseries
        
        
    def makeTable2(self,df=None,modeldict=None):
        if df is None:
            df=self.df
        if modeldict is None:
            modeldict=self.modeldict
        xvarlist=modeldict['xvars']
        table2path=self.helper.getname(os.path.join('print','table2.txt'))
        table2=f'created on{datetime.now()}\n\n'
        metric=['mean','std','min','max']
        descrip_list=[]
        binaries='\n\n\nBinaries\n\n'
        for xvar in xvarlist:
            if xvar[0:2]=='wq':
                xvar='secchi*'+xvar[2:]
            table2+=xvar+','
            for level in [0,1]: # pre and post-sandy
                descrip=df.loc[level][xvar].describe()
                descrip_list.append(descrip)
                for m in metric:
                    table2+=f'{round(descrip[m],1)},'
                if descrip['min'] in [0,1] and descrip['max'] in [0,1]:
                    if level==0:
                        binaries+=f'level{level},{xvar},{descrip["mean"]*descrip["count"]},'
                    else:binaries+=f'level{level},{xvar},{descrip["mean"]*descrip["count"]}\n'
                    
                    
            table2+='\n'
            
        table2+=binaries
            
        
        
        with open(table2path,'w') as f:
            f.write(table2)
            
        
        
    
        
    def printDFtoSumStats(self,df=None,varlist=None):
        pd.set_option('display.max_colwidth', None)
        
        if df is None:
            df=self.df
        if varlist is None:
            varlist=self.varlist
        
        levels=['Pre-Sandy','Post-Sandy']
        html_list=[]
        for level in df.index.levels[0]:
            html_list.append(levels[level]+'<br>'+df.loc[level].describe().to_html())
        
        self.sumstats_html='<br>'.join(html_list)
        printpath=self.helper.getname(os.path.join(self.printdir,'sumstats.html'))
        
        with open(printpath,'w') as f:
            f.write(self.sumstats_html) 
        
        
    def make2dHistogram(self,):   
        try:self.time_arraytup
        except: self.makeTimeListArrayList()
        timelist_arraylist,varlist=self.time_arraytup
        #varlist=self.varlist
        #t_idx=varlist.index('sale_year')
        timelenlist=[arraylist[0].shape[0] for arraylist in timelist_arraylist]
        
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
        self.logger.info(f'timelist_arraylist shapes:{[_array.shape for arraylist in timelist_arraylist for _array in arraylist]}')
        return timelist_arraylist,varlist
    
    
    def getCPI(self,to_year=2015,cpi_kwargs={}):
        #modeldict=self.modeldict
        '''kwargs={}
        if 'cpi-area' in cpi_kwargs:
            kwargs['area']=cpi_kwargs['cpi-area']
        if 'cpi-items' in cpi_kwargs:
            kwargs['items']=cpi_kwargs['cpi-items']'''
        cpi_kwargs['to']=to_year
        kwargstring=''.join([key+"-"+str(val)+'_' for key,val in cpi_kwargs.items()])
        stem=f'cpi_factors-{kwargstring}.json'
        excludechars=['\\', '/', ',',' ']
        stem=''.join([char for char in stem if char not in excludechars])
        cpi_factors_path=os.path.join(self.datadir,stem)
        try:
            with open(cpi_factors_path,'r') as f:
                cpi_factors=json.load(f)
            return cpi_factors
        except:
            self.logger.info('building cpi_factors')
        #import cpi
        cpi_factor_list=[]
        for t in self.timelist:#
            
            cpi_factor=np.float64(cpi.inflate(1,int(t),**cpi_kwargs))
            cpi_factor_list.append(cpi_factor)
        with open(cpi_factors_path,'w') as f:
            json.dump(cpi_factor_list,f)
        return cpi_factor_list
    
    def addRealByCPI(self,to_year=2015):
        try:self.time_arraytup
        except: self.makeTimeListArrayList()
        cpi_dollar_vars=self.modeldict['cpi_dollar_vars']    
        try:
            deflated_array_list=[]
            timelist_arraylist,varlist=self.time_arraytup
            for dollarvar,cpi_kwargs in cpi_dollar_vars.items():
                cpi_factor_list=self.getCPI(to_year=2015,cpi_kwargs=cpi_kwargs)
                for t in range(len(timelist_arraylist)):
                    nparraylist=timelist_arraylist[t]
                    var_idx=varlist.index(dollarvar)
                    cpi_factor=cpi_factor_list[t]
                    real_dollar_array=nparraylist[var_idx]*cpi_factor
                    nparraylist.append(real_dollar_array)
                    #nparray=np.concatenate([nparray,np.float64(nparray[:,var_idx][:,None])*cpi_factor],axis=1) 
                timelist_arraylist[t]=nparraylist
                newname=dollarvar+'_real-'+str(to_year)
                varlist.append(newname)
                #if newname[:4]!='sale':
                #    self.std_transform.append(newname) #these will be standardized
            self.logger.info(f'np.shape for timelist_arraylist:{[nparray.shape for arraylist in timelist_arraylist for nparray in arraylist]}')
            self.varlist=varlist
            self.time_arraytup_cpi=(timelist_arraylist,varlist)
            return
        except:
            self.logger.exception('')
    
            
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
