import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class IslandEffects:
    def __init__(self,):
        self.xlabel_rotate=65
    
    
    def makebigX(self,df):
        len0,len1=df.shape
        latlonstring_iloc_dict={}
        latlon_string_list=(df.loc[:,'latitude'].astype(str)+df.loc[:,'longitude'].astype(str)).to_list()
        last_unique_dict={}
        for i,latlon in enumerate(latlon_string_list):
            last_unique_dict[latlon]=i # keeps only last since earlier are overwritten.
        last_unique_rows=[val for key,val in last_unique_dict.items()]
        #print(f'bigX len(last_unique_rows):{len(last_unique_rows)}')
        unique_df=df.iloc[last_unique_rows].copy()
        self.bigX=unique_df
        return unique_df
            
    '''def makeCensusDict(self,df):
        censusvars=['education','income_real-2015','povertyleve','pct_white','pct_asian','pct_black']
        columnlist=[col for col in df.columns]
        censusdf=pd.DataFrame({'census_str':['']*len(df.index),index=df.index})
        for var in censusvars:
            censusdf.loc[slice(None),'census_str']+=df.loc[slice(None),var].astype(str)
        blockcount=0
        census_str_list=censusdf.loc[slice(None),'census_str'].to_list()
        yrlist=list(df.index.levels[1])
        censusdict={yr:}
        '''

    def addOmittedDistance(self,df,distancevars):
        last_dist=self.modeldict['distance'][-1]
        label=f'Distance to Shoreline {last_dist}m-4000m'
        if not label in df.columns:
            #df_idx=pd.Series([False]*len(df.index),index=df.index)
            df_idx=1-df.loc[:,distancevars].sum(axis=1)
            assert df_idx.max()==1, f' expecting 1 but df_idx.max():{df_idx.max()}'
            '''for var in distancevars: # loop through a bunch of 'or' operators to look for rows without a distance( incl bayfront/access).
                 #df_idx+=df.loc[slice(None),var]==1
                df_idx=df_idx|df.loc[slice(None),var]==1 # | (pipe) is or for dataframes'''
            df.loc[:,label]=df_idx
            distancevars.append(label)
        return df,distancevars
    
    def averageByDistance(self,df,distancevars,targetdf=None):
        if targetdf is None:
            targetdf=df.copy()
        #else: print(f'targetdf type: {type(targetdf)}')
        df,distancevars=self.addOmittedDistance(df,distancevars)
        #bigX_dist_avg_df=pd.DataFrame({col:[None]*len(distancevars)for col in df.columns},index=distancevars)
        bigX_dist_avg_df=pd.DataFrame()
        avg_data_list=[]
        for var in distancevars:
            df_idx=df.loc[slice(None),var]==1
            mean_df=targetdf.loc[df_idx].mean()
            keys=mean_df.keys()
            df_i=pd.DataFrame({key:mean_df.loc[key] for key in keys},index=[var])
            df_i.loc[:,'d_count']=df_idx.sum()
            avg_data_list.append(df_i)
        bigX_dist_avg_df=pd.concat(avg_data_list)
        return bigX_dist_avg_df
            #for key,val in meanvals.items()
   
    def setTimeDummies(self,df):
        regex_dv_y2k=re.compile('dv_20[0-9][0-9]')
        cols=list(df.columns)
        bigX_dist_avg_df_list=[df.copy(),df.copy()] # one for pre, one for post Sandy
        for var in cols:
            if re.search(regex_dv_y2k,var):
                bigX_dist_avg_df_list[0].loc[:,var]=1/11
                bigX_dist_avg_df_list[1].loc[:,var]=1/4
        return bigX_dist_avg_df_list
        
            


    def incrementWQ(self,df,incr):
        names=list(df.columns)
        wqnames=[var for idx,var in enumerate(names) if var[:6]=='secchi'] # *D necessary to distinguish from pre-existing distance bands
        #wqnames.extend(['secchi*bayfront', 'secchi*wateraccess'])
        df_plus=df.copy()
        for name in wqnames:
            df_idx=df_plus.loc[:,name]!=0 # so dummy var X WQ interaction zeros are not incremented
            df_plus.loc[df_idx,name]+=incr
        return df_plus
    
    '''
    def wtYfrombigX(self,bigX,yvar):
        
    
        return bigY_dist_wt_time_df
    '''
        
    def estimateWQEffects(self,wq_change_m=3.28084**-1,df=None,resultsdict=None):
        
        
        if df is None:
            df=self.df.copy()
        if resultsdict is None:
            modeltype='ols';periodlist=[0,1]
            resultsdict=self.retrieveLastResults(modeltype=modeltype,periodlist=periodlist)
        model=resultsdict[0]['results']
        names=list(model.name_x)
        distancevars=['bayfront','wateraccess']
        for name in names:
            if name[:21]=='Distance to Shoreline':
                distancevars.append(name)
        wq_dist_vars=[]
        for name in names:
            if name[:7]=='secchi*':# and not name[-11:]=='wateraccess':
            #if name[:7]=='secchi*' and not name[-11:]=='wateraccess':
                wq_dist_vars.append(name)
        wq_dist_vars.append('secchi')# to go with the omitted var at the end
        
        try:
            bigX=self.bigX
        except:
            bigX=self.makebigX(df) # condenses data across all times to latest observation at each lat/lon
        bigX_dist_avg_df=self.averageByDistance(bigX,distancevars)
        
        #bigY_dist_wt_time_df=self.wtYfrombigX(bigX,yvar) # went in diff direction
        bigX_dist_avg_df.loc[:,'CONSTANT']=1
        bigX_dist_avg_df_list=self.setTimeDummies(bigX_dist_avg_df)
        #bigX.index = pd.RangeIndex(len(bigX.index))
        #print(bigX_dist_avg_df)
        #self.bigX_dist_avg_df=bigX_dist_avg_df
        
        #effect_dict={}
        for period,result in resultsdict.items():
            
            model=result['results']
            modeldict=result['modeldict']
            yvar=modeldict['yvar']
            p=modeldict['period']
            bigX_dist_avg_df_p=bigX_dist_avg_df_list[p]
            names=list(model.name_x)
            #wq_var_idx=[names.index(var) for var in wq_dist_vars]
            x0=bigX_dist_avg_df_p.loc[:,names]
            #print('x0',x0)
            x1=self.incrementWQ(x0,wq_change_m)
            #print('x1',x1)
            betas=pd.DataFrame(list(model.betas),index=names)#np.array(model.betas).flatten()
            #print('betas',betas)
            wqbetas=betas.loc[wq_dist_vars]
            std_errs=pd.DataFrame(list(model.std_err),index=names)
            wq_std_errs=std_errs.loc[wq_dist_vars]
            #effect_dict['modeldict']=modeldict
            dwt=bigX_dist_avg_df_p.loc[:,'d_count'].to_numpy() # number of obs per distance band and access/bayfront
            dwt=dwt/dwt.sum() # as a share
            
            
            before=np.exp((x0@betas).to_numpy())
            marginal=before*wqbetas.to_numpy()
            #print('marginal',marginal)
            bigX_dist_avg_df_p.loc[:,f'marginal_p{p}']=marginal
            #print("bigX_dist_avg_df_p.loc[:,f'marginal_p{p}']",bigX_dist_avg_df_p.loc[:,f'marginal_p{p}'])
            l=(wqbetas-1.96*wq_std_errs).to_numpy()
            #print(f'l.shape:{l.shape},before.shape:{before.shape}')
            bigX_dist_avg_df_p.loc[:,f'lower95_marginal_p{p}']=before*l
            u=(wqbetas+1.96*wq_std_errs).to_numpy()
            bigX_dist_avg_df_p.loc[:,f'upper95_marginal_p{p}']=before*u
            #print('l',l)
            #print('u',u)
            bigX_dist_avg_df_p.loc[:,f'dwt_marginal_p{p}']=bigX_dist_avg_df_p.loc[:,f'marginal_p{p}']*dwt
            bigX_dist_avg_df_p.loc[:,f'dwt_lower95_marginal_p{p}']=bigX_dist_avg_df_p.loc[:,f'lower95_marginal_p{p}']*dwt
            bigX_dist_avg_df_p.loc[:,f'dwt_upper95_marginal_p{p}']=bigX_dist_avg_df_p.loc[:,f'upper95_marginal_p{p}']*dwt
            
            
            after=np.exp((x1@betas).to_numpy())
            bigX_dist_avg_df_p.loc[:,f'yhat_p{p}']=before
            bigX_dist_avg_df_p.loc[:,f'yhat_p{p}_wq+{wq_change_m}']=after
            bigX_dist_avg_df_p.loc[:,f'effect_p{p}']=(after-before)/wq_change_m
            bigX_dist_avg_df_p.loc[:,f'dwt_effect_p{p}']=bigX_dist_avg_df_p.loc[:,f'effect_p{p}']*dwt
            
            bigX_dist_avg_df_p.loc[:,'ydelta']=after-before
            ydelta_pct=((after-before)/before).flatten()
            bigX_dist_avg_df_p.loc[:,'ydelta_pct']=ydelta_pct #really per 1 not 100
            y=bigX_dist_avg_df_p.loc[:,yvar].to_numpy() ### dist averages cover normalizing by distance band, which can be reversed with dwt.
            ### but need to create more wt's to normalize by time.
            y1=y*(1+ydelta_pct)
            #print(f'(y.shape,y1.shape,ydelt_pct.shape):{(y.shape,y1.shape,ydelta_pct.shape)}')
            bigX_dist_avg_df_p.loc[:,'y1']=y1
            effect=y1-y
            bigX_dist_avg_df_p.loc[:,'effect']=effect
            rate=.03
            annual_effects=(bigX_dist_avg_df_p.loc[:,['effect']].copy()*rate).round(0)
            
            print(f'annual_dist_avg_effects for period-{p}, r={rate}\n')
            print(annual_effects)
            print(f'weighted grand average effect for p-{p}, r={rate},\n {annual_effects.T@dwt}')
            
            annual_effects.to_csv(os.path.join(self.resultsdir,f'avg_annual_effects_r-{rate}_p-{p}.csv'),header=['Average Treatment Effect ($)'])
            
            
            
            
            
        self.bigX_dist_avg_df=bigX_dist_avg_df
        self.bigX_dist_avg_df_list=bigX_dist_avg_df_list
        return bigX_dist_avg_df,bigX_dist_avg_df_list
    
    
    def estimateAnnualWQAvgMarginalEffect(self,):
        try:bigX_dist_avg_df_list=self.bigX_dist_avg_df_list # all dfs in list based on bigX, but with time dummy X's set for each period in list.
        except: 
            _,bigX_dist_avg_df_list=self.estimateWQEffects(effect=3.28084**-1) # a dataframe averaged within distance bands and with partial derivative and delta based marginal effects
        for p in [0,1]:
            df=bigX_dist_avg_df_list[p]
            sum_df=df.sum() # summing over the distance bands (and bayfront/access) in bigX_dist_avg_df's
            ame=sum_df[f'dwt_marginal_p{p}']
            print(f'avg marginal effect dwt sum for period{p} = {ame}')
            rate=0.03
            print(f'at r={rate}, annual benefits for period{p} = {ame*rate}')
            
            
            
    def getAnnualWQAvgEffect(self,wq_change_m=3.28084**-1,resultsdict=None):
        '''
        for non-marginal changes in wq
        '''
        if resultsdict is None:
            modeltype='ols';periodlist=[0,1]
            resultsdict=self.retrieveLastResults(modeltype=modeltype,periodlist=periodlist)
        
        #First calculate E[y1] and E[y2] then pct change for each row.
        
        try:
            bigX=self.bigX
        except:
            bigX=self.makebigX(self.df)
            
        bigX_tdummy_avg_list=self.setTimeDummies(bigX.copy())   
        
        for period,result in resultsdict.items():
            model=result['results']
            names=list(model.name_x)
            distancevars=['bayfront','wateraccess']
            for name in names:
                if name[:21]=='Distance to Shoreline':
                    distancevars.append(name)
            modeldict=result['modeldict']
            yvar=modeldict['yvar']
            p=modeldict['period']
            names=list(model.name_x)
            #wq_var_idx=[names.index(var) for var in wq_dist_vars]
            bigX_tdummy_avg=bigX_tdummy_avg_list[p]
            bigX_tdummy_avg.loc[slice(None),'CONSTANT']=1 # matches name of beta from estimator
            bigXplus_tdummy_avg=self.incrementWQ(bigX_tdummy_avg,wq_change_m)
        
            
            x0=bigX_tdummy_avg.loc[:,names]
            x1=bigXplus_tdummy_avg.loc[:,names]
            betas=np.array(model.betas).flatten()
            #wqbetas=betas[wq_var_idx]
            std_errs=np.array(model.std_err)
            yhat0=np.exp((x0@betas).to_numpy())
            yhat1=np.exp((x1@betas).to_numpy())
            #print(f'yhat0.shape,yhat1.shape:{(yhat0.shape,yhat1.shape)}')
            pct_ch=(yhat1-yhat0)/yhat0 # except without the 100% exp(s2/2) not calculated b/c cancels out.
            
            pct_ch_df=pd.DataFrame(pct_ch,index=bigX.index)
            avg_pct_ch=self.averageByDistance(bigX,distancevars,
                                                    targetdf=pct_ch_df)
            #print('avg_pct_ch',avg_pct_ch)
            #print('pct_ch',pct_ch)
            #now calculate y1-y0=y0[1+pct_ch] - y0 
            bigY0=bigX.loc[:,yvar].copy()
            bigY1=bigY0*(1+pct_ch)
            effect=bigY1-bigY0
            effect_name=f'effect_p{p}'
            bigX_tdummy_avg.loc[:,effect_name]=effect
            rate=.03
            average_effect=effect.mean()
            #print(f'(bigY0.shape,bigY1.shape,effect.shape,pct_ch.shape):{(bigY0.shape,bigY1.shape,effect.shape,pct_ch.shape)}')
            print(f'grand average effect period-{p}, {average_effect} at r={rate}, {average_effect*rate}')
            dist_avg_effects=self.averageByDistance(bigX,distancevars,
                                                    targetdf=bigX_tdummy_avg.loc[:,[effect_name,]])
            self.bigX_tdummy_avg_list=bigX_tdummy_avg_list
            annual_dist_avg_effects=dist_avg_effects*.03
            print(f'dist_avg_effects for period-{p}',dist_avg_effects)
            print(f'annual_dist_avg_effects for period-{p}, r={rate}',annual_dist_avg_effects)

                
        
            
    
    def createEffectsGraph(self,drop_wateraccess=False):
        try: bigX_dist_avg_df=self.bigX_dist_avg_df,bigX_dist_avg_df_list=self.bigX_dist_avg_df_list
        except: pl
            self.logger.exception(f'creating new bigX_dist_avg_df and list')
            bigX_dist_avg_df,bigX_dist_avg_df_list=self.estimateWQEffects(wq_change_m=3.28084**-1) # a dataframe averaged within distance bands and with partial derivative and delta based marginal effects
        if drop_wateraccess:
            bigX_dist_avg_df.drop(index='wateraccess',inplace=True)
            [df.drop(index='wateraccess',inplace=True) for df in bigX_dist_avg_df_list]
        """else:
            df_idx=bigX_dist_avg_df.index
            wa_idx=list(df_idx).index('wateraccess')
            new_idx=list(df_idx)
            wa=new_idx.pop(wa_idx)
            new_idx.insert(2,wa)
            print('new_idx',new_idx)
            bigX_dist_avg_df=bigX_dist_avg_df.reindex(new_idx,copy=True)
            bigX_dist_avg_df_list=[df.reindex(new_idx,copy=True) for df in bigX_dist_avg_df_list]"""
        x=list(bigX_dist_avg_df.index)
        print(x)
        effect_name=[]
        for var in x: # a loop for shortening names
            if var[:8].lower()=='distance':
                effect_name.append(var[22:])
            else:
                effect_name.append(var)
        
        fig=plt.figure(dpi=600,figsize=[10,6])
        #plt.xticks(rotation=self.xlabel_rotate)
        ax=fig.add_subplot()
        ax.set_title('Marginal Effects for Water Clarity by Distance Band')#Fixed Effects Estimates for Water Clarity by Distance from Shore Band')
        ax.set_xlabel('Distance from Shore Bands (not to scale)')
        ax.set_ylabel('Partial Derivatives of Sale Price by Water Clarity ($/m)')
        
        for p in [0,1]:
            bigX_dist_avg_df_p=bigX_dist_avg_df_list[p]
            #print('bigX_dist_avg_df_p',bigX_dist_avg_df_p)
            effect=bigX_dist_avg_df_p[f'marginal_p{p}'].to_numpy(dtype=np.float64)
            lower=bigX_dist_avg_df_p[f'lower95_marginal_p{p}'].to_numpy(dtype=np.float64)
            upper=bigX_dist_avg_df_p[f'upper95_marginal_p{p}'].to_numpy(dtype=np.float64)
            #print('effect',effect)
            #print('lower',lower)
            #print('upper',upper)
            self.makePlotWithCI(effect_name,effect,None,ax,**self.plot_dict_list[p],lower=lower,upper=upper)
        ax.legend(loc=1)
        ax.margins(0)
        plt.draw()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=65, ha='right')
        fig.tight_layout()
        #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.f'))
        figpath=self.helper.getname(os.path.join(self.printdir,'wq_effects_graph.png'))
        
        fig.savefig(figpath)
    
    def retrieveLastResults(self,modeltype='ols',periodlist=[0,1]):
        try: 
            assert self.resultsdictlist,"results not loaded, loading results"
            resultsdictlist=self.resultsdictlist
        except: 
            resultsdictlist=self.saveModelResults([],load=1)
        
        
        resultsdictflatlist=self.flattenListList(resultsdictlist)
        outdict={key:None for key in periodlist}
        i=-1
        while not all([outdict[key] for key in outdict]):
            i+=1
            resultsdict=resultsdictflatlist[i]
            modeldict=resultsdict['modeldict']
            p=modeldict['period']
            m=modeldict['modeltype']
            if m.lower()==modeltype:
                for key,val in outdict.items():
                    if not val:
                        if p==key:
                            outdict[key]=resultsdict
        return outdict
                    
    def createWQGraph(self,modeltype='ols',drop_wateraccess=False,drop_secchi=True):
        lastresults=self.retrieveLastResults(modeltype=modeltype,periodlist=[0,1])
        
        fig=plt.figure(dpi=600,figsize=[10,6])
        #plt.xticks(rotation=self.xlabel_rotate)
        #fig.autofmt_xdate(rotation=45)
        ax=fig.add_subplot()
        ax.set_title('Fixed Effects Estimates of Water Quality Coefficients')#Fixed Effects Estimates for Water Clarity by Distance from Shore Band')
        ax.set_xlabel('Distance from Shore Bands (not to scale)')
        ax.set_ylabel('OLS Coefficients for Water Clarity')
        #next part is not yet flexible due to color/hatch/linestyle(ls)
        
        for p in [0,1]:
            self.extractAndPlotWQ(lastresults[p],ax,drop_wateraccess=drop_wateraccess,drop_secchi=drop_secchi,**self.plot_dict_list[p])
        ax.legend(loc=1)
        ax.margins(0)
        plt.draw()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=65,ha='right')
        fig.tight_layout()
        figpath=self.helper.getname(os.path.join(self.printdir,'wq_graph.png'))
        fig.savefig(figpath)
        
    
    def extractAndPlotWQ(self,resultsdict,ax,plottitle='',color='b',hatch='x',ls='-',drop_wateraccess=False,drop_secchi=False):
        results=resultsdict['results']
        #modeldict=resultsdict['modeldict']
        #xvarlist=modeldict['xvars']
        if drop_wateraccess:
            water_str='wateraccess'
        else: water_str=''
        if drop_secchi:
            secchi_str='secchi\*'
        else:
            secchi_str='secchi'
        wqvars_idx,wqvars=zip(*[[idx,var] for idx,var in enumerate(results.name_x) if re.search(secchi_str,var.lower()) and var.lower()[-11:]!=water_str])
        wqvars_idx=list(wqvars_idx)
        wqvars=list(wqvars)
        if not drop_secchi:
            secchi_idx=wqvars.index('secchi')
            svar=wqvars.pop(secchi_idx)
            wqvars.append(svar)
            sidx=wqvars_idx.pop(secchi_idx)
            wqvars_idx.append(sidx)
        """if not drop_wateraccess:
            wa_idx=wqvars.index('secchi*wateraccess')
            wa=wqvars.pop(wa_idx)
            wqvars.insert(2,wa)
            wa=wqvars_idx.pop(wa_idx)
            wqvars_idx.insert(2,wa)"""
            
            
            
        
        '''for idx,var in enumerate(results.name_x):
            if var.lower()=='secchi':
                wqvars_idx.append(idx)
                wqvars.append('3200m-4000m')
                secchi=results.betas[wqvars_idx][0]*100'''
        #print(wqvars_idx,wqvars)
        #print('betas',results.betas)
        #print('std_err',results.std_err)
        
        wqcoefs=[results.betas[idx][0] for idx in wqvars_idx] # b/c pysal returns each beta inside its own list. *100 b/c pct. 
        wqcoef_stderrs=[results.std_err[idx] for idx in wqvars_idx] # 100 b/c pct.
        wqcoef_names=[]
        for var in wqvars: # a loop for shortening names
            if re.search('secchi\*distance',var.lower()):
                wqcoef_names.append(var[29:])
            elif re.search('secchi\*',var.lower()):
                wqcoef_names.append(var[7:])
            else:
                wqcoef_names.append(var)
        '''bcount=len(wqcoefs)
        for b in range(bcount-1):# add global constant to all terms except last one, which was the omitted variable
            wqcoefs[b]=wqcoefs[b]+wqcoefs[-1]
            wqcoef_stderrs[b]=wqcoef_stderrs[b]+wqcoef_stderrs[-1]'''
        self.logger.info(f'[wqcoef_names,wqcoefs,wqcoef_stderrs ]:{[wqcoef_names,wqcoefs,wqcoef_stderrs ]}')
        self.makePlotWithCI(wqcoef_names,wqcoefs,wqcoef_stderrs,ax,plottitle=plottitle,color=color,hatch=hatch,ls=ls)
        