import os
import numpy as np
import data_geo as dg
import pysal
import libpysal
#import geopandas
import geopy
import logging
import joblib
import pickle
from haversine import haversine_vector, Unit
#from scipy.sparse import dok_matrix

class SpatialModel():
    def __init__(self,modeldict=None):
        self.logger = logging.getLogger(__name__)
        if modeldict is None:
            modeldict={
                'combine_pre_post':0,
                'modeltype':'SEM',
                'nneighbor':[10,15,20],
                'crs':'epsg:4326'
            }
        self.modeldict=modeldict
        
    
        
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
        nn=self.modeldict['nneighbor']
        if type(nn) is int:
            klist=[nn]
        elif type(nn) is list:
            klist=nn
        else:assert False,f'halt, unrecongized nn:{nn}'
            
        if df is None:
            try:self.df
            except: self.arrayListToPandasDF()
            df=self.df
        
        if self.modeldict['combine_pre_post']==1:
            df_idx_0_list=[slice(None)]
        else:
            df_idx_0_list=[0,1]
        wtlistlist=[];resultslistlist=[]
        for idx0 in df_idx_0_list:
            dfi=df.loc[idx0]
            #print('selecting first 200 obs only')
            #dfi=dfi.iloc[:200,:]
            #gdfi=self.buildGeoDF(df=dfi)
            
            wtlist=self.makeInverseDistanceWeights(dfi,klist=klist)
            wtlistlist.append(wtlist)
            resultslistlist.append([self.runSpatialErrorModel(dfi,w,nn=k) for w,k in zip(wtlist,klist)])
        return resultslistlist
    
    def runSpatialErrorModel(self,df,w,nn=None):
        yvar='saleprice_real-2015'
        y=np.log10(df.loc[:][yvar].to_numpy())[:,None]#make 2 dimensionsl for spreg
        xvarlist=['secchi','totalbathroomsedited']
        '''xvarlist=[
            'secchi','bayfront','wateraccess','wqbayfront','wqwateraccess',
            'totalbathroomsedited','totallivingarea','saleacres',
            'distance_park','distance_nyc','distance_golf',
            'shorelinedistancedv3_1000','shorelinedistancedv1000_2000',
            'shorelinedistancedv2000_3000','shorelinedistancedv3000_4000',
            'wqshorelinedistancedv3_1000','wqshorelinedistancedv1000_2000',
            'wqshorelinedistancedv2000_3000','wqshorelinedistancedv3000_4000',
            'education','income_real-2015','povertylevel','pct_white']'''
                  
        #xvarlist=[varlist.pop(varlist.index(var)) for var in excludevars]
        
        x=df.loc[:][xvarlist].to_numpy()
        sem=pysal.model.spreg.ML_Error(y,x,w,name_y=yvar,name_x=xvarlist,name_w=f'inv_dist_nn{nn}')
        print(f'sem.name_x:{sem.name_x}')
        print(f'sem.betas:{sem.betas}')
        print(f'sem.std_err:{sem.std_err}')
        print(f'sem.z_stat:{sem.z_stat}')
        return sem
    
    def checkForWList(self,df):
        thehash=joblib.hash(df.to_csv().encode('utf-8'))
        path=os.path.join('data',thehash+'.pickle')
        if os.path.exists(path):
            try:
                with open(path,'rb') as f:
                    Wlist=pickle.load(f)
                self.logger.info(f'wlist retrieved from path:{path}')
                return Wlist
            except:
                self.logger.exception('path exists, but could not open wlist at path:{path}')
                return None
        else:
            return None
    
    def saveWList(self,Wlist,df):
        thehash=joblib.hash(df.to_csv().encode('utf-8'))
        path=os.path.join('data',thehash+'.pickle')
        with open(path,'wb') as f:
            pickle.dump(Wlist,f)
        self.logger.info(f'wlist saved to path:{path}')
        return
    
    def makeInverseDistanceWeights(self,dfi,klist=[20],distmat=1):
        '''
        from libpysal.weights import W
        neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
        weights = {0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}
        w = W(neighbors, weights)
        '''
        n=dfi.shape[0]
        savedWlist=self.checkForWList(dfi)
        if savedWlist:
            return savedWlist
        timelist=dfi.index.levels[0]
        neighbors_dictlist=[{} for _ in range(len(klist))]
        weights_dictlist=[{} for _ in range(len(klist))]
        if distmat:
            distmat=self.makeDistMat(dfi)
            for idx in range(n):
                dist,j_idx=zip(*sorted(zip(distmat[idx],list(range(n)))))
                dist=list(dist)
                j_idx=list(j_idx)
                dist,j_idx=self.removeOwnDist(idx,dist,j_idx)
                dist,j_idx=self.remove0Dist(dist,j_idx)
                for k_idx in range(len(klist)):
                    neighbors_dictlist[k_idx][idx]=j_idx[:klist[k_idx]]
                    weights_dictlist[k_idx][idx]=[distij**-1 for distij in dist[:klist[k_idx]]]
        else:
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
        for k_idx in range(len(klist)):
            self.logger.info(f'neighbors and weights: {neighbors_dictlist[k_idx],weights_dictlist[k_idx]}')
            wlist.append(libpysal.weights.W(neighbors_dictlist[k_idx],weights_dictlist[k_idx]))
            self.wlist=wlist
        self.saveWList(wlist,dfi)
        return wlist

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
            
            
        
        
                           
    def makeDistMat(self,df):
        lon_array=df['longitude'].to_numpy()
        lat_array=df['latitude'].to_numpy()
        n=lon_array.size
        pointarray=np.array([(lat_array[i],lon_array[i]) for i in range(n)],dtype=np.float16)
        istack=np.repeat(pointarray,repeats=n,axis=0)
        jstack=np.repeat(np.expand_dims(pointarray,0),axis=0,repeats=n).reshape([n**2,2],order='C')
        distmat=haversine_vector(istack,jstack).reshape(n,n)
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
                