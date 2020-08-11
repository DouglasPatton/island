import os
os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=4
import numpy as np
import data_geo as dg
from pysal import model as pysal_model
from libpysal import weights as libpysal_weights
#import geopandas
#import geopy
import logging
import joblib
import pickle
from haversine import haversine_vector, Unit
import re
from copy import deepcopy
from datetime import datetime
from statsmodels.regression.linear_model import OLS
#from scipy.sparse import dok_matrix

class Model():
    '''
    Formerly SpatialModel
    '''
    def __init__(self,modeldict):
        self.logger = logging.getLogger(__name__)
        self.modeldict=modeldict
        self.savespath=os.path.join(os.getcwd(),'saves')
        if not os.path.exists(self.savespath):os.makedirs(self.savespath)
    
        
    def buildGeoDF(self,df=None):
        if df is None:
            try:self.df
            except: self.arrayListToPandasDF()
            df=self.df
        try:
            points=geopandas.points_from_xy(df['longitude'],df['latitude'])
        except:
            points=geopandas.points_from_xy(df[:]['longitude'],df[:]['latitude'])
        gdf=geopandas.GeoDataFrame(df,geometry=points,crs={'init':self.crs})
        return gdf         
                 
    
        
    def run(self,df=None):
        # https://pysal.org/libpysal/generated/libpysal.weights.W.html#libpysal.weights.W
        modeldict=self.modeldict
        if re.search('statsmodels',modeldict['modeltype']):
            pass
            
        else:
            nn=self.modeldict['klist']
            if type(nn) is int:
                klist=[nn]
            elif type(nn) is list:
                klist=nn
            else:assert False,f'halt, unrecongized nn:{nn}'
            if not re.search('nn',modeldict['wt_type'].lower()):
                klist=['all']


            if df is None:
                try:self.df
                except: self.arrayListToPandasDF()
                df=self.df
                assert False, 'broken, no distancevars'

            if self.modeldict['combine_pre_post']==1:
                df_idx_0_list=[slice(None)]
            else:
                df_idx_0_list=[0,1]
            wtlistlist=[];resultsdictlist=[]
            modeldict_i=deepcopy(modeldict) 
            for idx0 in df_idx_0_list:
                dfi=df.loc[idx0]
                modeldict_i['period']=idx0
                wtlist=self.makeInverseDistanceWeights(dfi,modeldict=modeldict_i)

                wtlistlist.append(wtlist)
                for w,k in zip(wtlist,klist):
                    modeldict_i_k=deepcopy(modeldict_i) 
                    modeldict_i_k['klist']=k
                    resultsdict=self.runPysalModel(dfi,w,nn=k,t=idx0,modeldict=modeldict_i_k)
                    resultsdictlist.append(resultsdict)
            return resultsdictlist

    
    def prepModelData(self,df,w=None,nn=None,t=None,modeldict=None):
        if modeldict is None: modeldict=self.modeldict
        
        
        yvar=modeldict['yvar']
        transform_dict=modeldict['transform']
        do_log_y=transform_dict['ln_y']
        do_log_wq=transform_dict['ln_wq']
        y=df.loc[:][yvar].to_numpy(dtype=np.float32)[:,None]#make y 2 dimensions for spreg
        if do_log_y: y=np.log(y)
        xvarlist=modeldict['xvars']
        preSdropyears=[i for i in range(2013,2016)]# 2015+1 b/c python
        postSdropyears=[i for i in range(2002,2012)]# 2011+1 b/c python
        dropyearslist=[preSdropyears,postSdropyears]
        if t in [0,1]:
            dropyears=dropyearslist[t]     
            if t:
                dropyears.append('2015') #drop the excluded dummy too
            else:
                dropyears.append('2003')
        else: assert False, 'halt, not developed'
        dropyear_dvs=[f'dv_{i}' for i in dropyears]
        [xvarlist.pop(xvarlist.index(var)) for var in dropyear_dvs]
        #df,xvarlist=self.doDistanceVars(df,xvarlist,modeldict)
        
        
        modeldict['xvars']=xvarlist
        self.logger.info(f'xvarlist:{xvarlist}')
        wqvar_idx_list=[idx for idx,var in enumerate(xvarlist) if re.search('wq',var) or re.search('secchi',var)]
        
        
        x=df.loc[:][xvarlist].to_numpy(dtype=np.float64)
        if do_log_wq:
            for idx in wqvar_idx_list:
                self.logger.info(f'LogPos transform of variable: {xvarlist[idx]}')
                x=self.myLogPos(x,col=idx)
            
        self.logger.info(f'y:{y}')
        self.logger.info(f'x.shape:{x.shape}')
        self.logger.info(f'x:{x}')

        args=[y,x,w]
        kwargs={'name_y':yvar,'name_x':xvarlist}
        try:
            wt_type=modeldict['wt_type']
            wt_norm=modeldict['wt_norm']
            NNscope=modeldict['NNscope']
            name_w=f'{wt_type}-{wt_norm}-{NNscope}-{nn}'
            kwargs['name_w']=name_w
        except:
            self.logger.exception(f'no info found for weights when making kwargs')
            
        
        return args,kwargs
    
    
    def runPysalModel(self,df,w,nn=None,t=None,modeldict=None):
        args,kwargs=self.prepModelData(df,w,nn=nn,t=t,modeldict=modeldict)
        modeltype=modeldict['modeltype']
        if modeltype.lower()=='sem':
            estimator=pysal_model.spreg.ML_Error
        elif modeltype.lower()=='slm':
            estimator=pysal_model.spreg.ML_Lag
        elif modeltype.lower()=='ols':
            kwargs['spat_diag']=True
            estimator=pysal_model.spreg.OLS
        elif modeltype.lower()=='gm_error_het':
            estimator=pysal_model.spreg.GM_Error_Het # https://spreg.readthedocs.io/en/latest/generated/spreg.GM_Error_Het.html#spreg.GM_Error_Het
            kwargs['w']=args.pop(-1) #  move arg to kwarg for these estimators
        elif modeltype.lower()=='gm_combo_het':
            estimator=pysal_model.spreg.GM_Combo_Het
            kwargs['w']=args.pop(-1)
        elif modeltype.lower()=='gm_lag':
            estimator=pysal_model.spreg.GM_Lag
            kwargs['w']=args.pop(-1)
                
        hashable_key=[args[:2],{key:val for key,val in kwargs.items() if key!='w'}] # w probably not hashable, so exclude
        model_filestring=f'_model_{modeltype}'
        trysavedmodel=self.checkForSaveHash(hashable_key,filestring=model_filestring)
        #self.logparams(modeldict)
        if trysavedmodel:
            self.logger.warning(f'hashkeysaved model model already exists for modeldict:{modeldict}, skipping')
            model=trysavedmodel         
        else:
            model=estimator(*args,**kwargs) # this is it!
            
            self.saveByHashID(hashable_key,model,filestring=model_filestring)
        try:
            #print(model.summary)
            self.logger.info(model.summary)
        except:
            self.logger.exception('could not print summary')
            
        resultsdict={'modeldict':deepcopy(modeldict),'results':model}
        return resultsdict
    
    def logparams(self,modeldict,recursivekey=''):#for mlflow, which has been removed
        for key,val in modeldict.items():
            if recursivekey:
                key=recursivekey+'..'+key
            if not type(val) is dict:
                mlflow.log_param(key,val)
            else:
                self.logparams(val,recursivekey=key)
    
    
    def myLogPos(self,x,col=None):
        if col is None:
            col=slice(None)
        x[:,col][x[:,col]>0]=np.log(x[:,col][x[:,col]>0])
        return x
    
    
    def checkForWList(self,df,modeldict):
        hashable_key=[df.to_csv().encode('utf-8'),modeldict]
        Wlist=self.checkForSaveHash(hashable_key,filestring='_Wlist')
        return Wlist
    
    
    def saveWList(self,Wlist,df,modeldict):
        hashable_key=[df.to_csv().encode('utf-8'),modeldict]
        self.saveByHashID(hashable_key,Wlist,filestring="_Wlist")
        
    def checkForSaveHash(self,hashable_key,filestring=""):
        thehash=joblib.hash(hashable_key)
        path=os.path.join(self.savespath,thehash+filestring+'.pickle')
        if os.path.exists(path):
            try:
                with open(path,'rb') as f:
                    thing=pickle.load(f)
                self.logger.info(f'thing retrieved from path:{path}')
                return thing
            except:
                self.logger.exception(f'path exists, but could not open thing at path:{path}')
                return None
        else:
            self.logger.info(f'no file found for hash:{thehash} at path:{path}')
            return None

        
    def saveByHashID(self,hashable_key,thing,filestring=""):
        thehash=joblib.hash(hashable_key)
        path=os.path.join(self.savespath,thehash+filestring+'.pickle')
        keypath=path[:-7]+'_hashable-key.pickle'
        with open(path,'wb') as f:
            pickle.dump(thing,f)
        with open(keypath,'wb') as f:
            pickle.dump(hashable_key,f)
        self.logger.info(f'filestring saved to path:{path}')
        return
    
    def neighbornetwork(self,w):
        
        for i in range(3):
            neighblist=w.neighbors[i]
            print(f'{i} has {neighblist}')
            for j in neighblist:
                neighblist2=w.neighbors[j]
                print(f'{j} has {neighblist2}')
                for k in neighblist2:
                    neighblist3=w.neighbors[k]
                    print(f'{k} has {neighblist3}')
                
            
            
    
    
    def makeInverseDistanceWeights(self,dfi,modeldict=None,skipW=0,):
        '''
        from libpysal.weights import W
        neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
        weights = {0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}
        w = W(neighbors, weights)
        '''
        try:
            distmat=modeldict['distmat']
        except:
            distmat=1
        try:
            klist=modeldict['klist']
        except:
            klist=None
        wt_type=modeldict['wt_type']
        wt_norm=modeldict['wt_norm']
        NNscope=modeldict['NNscope']
        klist=modeldict['klist']
        if not type(klist) is list:
            klist=[klist]
            
        is_inv_dist=re.search('inverse_distance',wt_type.lower())
        if is_inv_dist:
            exp_srch=re.search('exp',wt_type)
            if exp_srch:
                exp=float(wt_type[exp_srch.end():])

            else:
                exp=1
        is_nn=re.search('nn',wt_type.lower())
            
        wt_modeldict={key:modeldict[key] for key in ['wt_type','wt_norm','NNscope','klist','combine_pre_post']}
        
        savedWlist=self.checkForWList(dfi,wt_modeldict)
        if savedWlist:
            #print(self.neighbornetwork(savedWlist[-1]))
            return savedWlist
        timelist=dfi.index.remove_unused_levels().levels[0]#dfi.index.levels[0]
        neighbors_dictlist=[{} for _ in range(len(klist))]
        weights_dictlist=[{} for _ in range(len(klist))]
        if NNscope=='year':
            index1=timelist
        if NNscope=='period':
            index1=[slice(None)]
            
        if distmat:
            n_cum=0
            for scope_idx in index1:
                dfi_s=dfi.loc[scope_idx]
                n=dfi_s.shape[0]
                distmat=self.makeDistMat(dfi_s)
                #zeroblocker=np.array([[10**20]],dtype=np.float32).repeat(n,1).repeat(n,0)
                #distmat[distmat==0]=zeroblocker[distmat==0]
                distmat[distmat==0]=10**20# assuming only one's self can be zero distance away, also treats self if sold more than once
                idxarray=np.arange(n_cum,n_cum+n)
                n_cum+=n # for the next iteration
                sortidx=distmat.argsort(axis=-1) # default is axis=-1, sorting across the 'columns' of the 2d array, so each row is sorted. 
                self.logger.info(f'making weights with weight type:{wt_type}')
                for pos,idx in enumerate(idxarray):
                    sortvec=sortidx[pos,:]
                    for k_idx in range(len(klist)):
                        if is_nn:
                            neighb_list=list(idxarray[sortvec[:klist[k_idx]]])
                        else:
                            neighb_list=list(idxarray[sortvec])
                        if is_inv_dist:
                            if is_nn:
                                inv_wt_arr=distmat[pos,sortvec[:klist[k_idx]]]**-exp
                            else:
                                dist_arr=distmat[pos,sortvec]
                                inv_wt_arr=dist_arr**-exp
                                inv_wt_arr[dist_arr>10**19]=0
                            
                            weight_list=list(inv_wt_arr)
                            '''try:
                                selfpos=neighb_list.index(idx)
                            except:
                                selfpos=None
                            if selfpos:
                                print(f'selfpos:{selfpos},weight_list[selfpos]:{weight_list[selfpos]}')'''
                                
                        elif wt_type.lower()=='nn':
                            weight_list=[1 for _ in range(klist[k_idx])]
                        
                        if wt_norm:
                            wt_arr=np.array(weight_list, dtype=np.float32)
                            if wt_norm=='rowmax':
                                norm_wt_arr=wt_arr/wt_arr.max(axis=-1,keepdims=1)
                            elif wt_norm=='rowsum':
                                norm_wt_arr=wt_arr/wt_arr.sum(axis=-1,keepdims=1)
                            elif wt_norm=='doublesum':
                                norm_wt_arr=wt_arr/wt_arr.sum(keepdims=1)
                            weight_list=list(norm_wt_arr)
                        


                        neighbors_dictlist[k_idx][idx]=neighb_list #notably idx not pos since idx is cumulative across time
                        weights_dictlist[k_idx][idx]=weight_list
        else:
            assert False,'code needs updating for NNscope'
            #for year in self.
            pointlist=self.makePointListFromDF(dfi)
            for idx in range(n):
                self.logger.info(f'starting distances for idx/n-1:{idx}/{n-1}')
                distlist=self.makeDistList(idx,pointlist)
                dist,j_idx=zip(*sorted(zip(distlist,list(range(n))))) 
                self.logger.info(f'for idx:{idx}, dist[0:5]:{dist[0:5]} for j_idx[0:5]:{j_idx[0:5]}')
                #    j_idx indexes positions of other points, dist is their distance
                dist=list(dist)
                j_idx=list(j_idx)
                dist,j_idx=self.removeOwnDist(idx,dist,j_idx)
                for k_idx in range(len(klist)):
                    nn_stop_idx=klist[k_idx]
                    neighbors_dictlist[k_idx][idx]=j_idx[:nn_stop_idx]
                    weights_dictlist[k_idx][idx]=[dist_j**(-1) for dist_j in dist[:nn_stop_idx]]
        #wlist=[]
        #for k_idx in range(len(klist)):
        #    wlist.append(self.makeSparseMat(neighbors_dictlist[k_idx],weights_dictlist[k_idx],n))
        
        wlist=[]
        self.neighborsandweights=(neighbors_dictlist,weights_dictlist)
        if skipW:
            return self.neighborsandweights
        for k_idx in range(len(klist)):
            #self.logger.info(f'neighbors and weights: {neighbors_dictlist[k_idx],weights_dictlist[k_idx]}')
            wlist.append(libpysal_weights.W(neighbors_dictlist[k_idx],weights_dictlist[k_idx]))
        self.wlist=wlist
        self.saveWList(wlist,dfi,wt_modeldict)
        return wlist
    
                             
    def makeDistMat(self,df):
        lon_array=df['longitude'].to_numpy()
        lat_array=df['latitude'].to_numpy()
        n=lon_array.size
        pointarray=np.concatenate([lat_array[:,None],lon_array[:,None]],axis=1)
        np.array(pointarray, dtype=np.float32)
        istack=np.repeat(pointarray,repeats=n,axis=0)
        jstack=np.repeat(np.expand_dims(pointarray,0),axis=0,repeats=n).reshape([n**2,2],order='C')
        distmat=haversine_vector(istack,jstack).reshape(n,n)
        #self.logger.info(f'lon_array[:5],lat_array[:5],distmat[0:5,0:5]:{[lon_array[:5],lat_array[:5],distmat[0:5,0:5]]}')
        '''for i in range(n):
            ilatlon=pointarray[i,:]
            ilatlonstack=np.repeat(ilatlon,n)
            distmat[i,:]= #(lat,lon)
        pointlist=[(lon_array[i],lat_array[i]) for i in range(n)]
        distmat=np.empty([n,n],dtype=np.float32) #
        for i0 in range(n):
            for i1 in range(i0+1,n):
                distance=geopy.distance.great_circle(pointlist[i0],pointlist[i1]).meters #.geodesic is more accurate but slower
                distmat[i0,i1]=distance#not taking advantage of symmetry of distance for storage, assuming ample ram
                distmat[i1,i0]=distance'''
        self.distmat=distmat
        return distmat
    

    def makeSparseMat(self,neighb_dict,weight_dict,n):
        sparse_mat=dok_matrix((n,n),dtype=np.float32)
        for i,jlist in neighb_dict.items():
            for j_idx,j in enumerate(jlist):
                sparse_mat[i,j]=weight_dict[i][j_idx]
        return sparse_mat
    def remove0Dist(self,dist,j_idx):
        droplist=[]
        for d_idx,d in enumerate(dist):
            if d==0:
                droplist.append(d_idx)
        for d_idx in droplist[::-1]:#start on rhs of list so indices don't change for next pop
            dist.pop(d_idx)
            j_idx.pop(d_idx)
        return (dist,j_idx)
    
    def removeOwnDist(self,idx,dist,j_idx):
        iloc=j_idx.index(idx)
        j_idx.pop(iloc)
        dist.pop(iloc)
        return (dist,j_idx)
    
    
    def makePointListFromDF(self,df):
        lon_array=df['longitude'].to_numpy()
        lat_array=df['latitude'].to_numpy()
        n=lon_array.size
        pointlist=[(lon_array[i],lat_array[i]) for i in range(n)]
        return pointlist
   
    def makeDistList(self,idx,pointlist):
        otherpoints=pointlist.copy()
        thispoint=otherpoints[idx]
        distlist=[[10**299] for _ in range(len(otherpoints))]   
        for p_idx in range(len(otherpoints)):
            if p_idx!=idx:
                thatpoint=otherpoints[p_idx]
                distlist[p_idx]=geopy.distance.great_circle(thispoint,thatpoint).meters/1000
            else:
                distlist[idx]=0
        return distlist
            
  
                
